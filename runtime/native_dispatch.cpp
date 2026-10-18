// Same ATen operations and expert order as ExpertCache.apply, in one native call.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "libtorch/linear.h"
#include "grouped_dispatch.h"

at::Tensor grouped_linear(const at::Tensor& x,
    const std::vector<std::shared_ptr<BC_LinearEXL3>>& layers)
{
    c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous()
        && x.scalar_type() == at::kHalf, "Expected contiguous fp16 CUDA rows");
    TORCH_CHECK(!layers.empty() && layers.size() <= 12, "Expected 1..12 projections");
    const int e = layers.size(), k = x.size(1), n = layers[0]->svh.numel();
    const int bits = layers[0]->K;
    TORCH_CHECK(x.size(0) == 1 || x.size(0) == e, "Expected broadcast or one row per projection");
    TORCH_CHECK((bits == 2 || bits == 3) && k % 128 == 0 && n % 128 == 0, "Unsupported geometry");
    GroupedPointers ptrs{};
    ptrs.count = e;
    for (int i = 0; i < e; ++i) {
        const auto& p = layers[i];
        TORCH_CHECK(p->K == bits && p->mul1 && !p->mcg && !p->bias
            && p->suh.numel() == k && p->svh.numel() == n,
            "Grouped dispatch requires homogeneous MUL1 K2/K3 projections without bias");
        TORCH_CHECK(p->trellis.device() == x.device() && p->suh.device() == x.device()
            && p->svh.device() == x.device(), "Projection device mismatch");
        ptrs.trellis[i] = reinterpret_cast<const uint16_t*>(p->trellis.data_ptr<int16_t>());
        ptrs.suh[i] = reinterpret_cast<const half*>(p->suh.data_ptr<at::Half>());
        ptrs.svh[i] = reinterpret_cast<const half*>(p->svh.data_ptr<at::Half>());
    }
    auto out = at::empty({e, n}, x.options().dtype(at::kFloat));
    auto had = at::empty({e, k}, x.options());
    launch_grouped(x.data_ptr(), out.data_ptr(), had.data_ptr(), k, n,
        x.size(0) == 1 ? 0 : k, bits, ptrs);
    return out;
}

at::Tensor grouped_down(const at::Tensor& x,
    const std::vector<std::shared_ptr<BC_LinearEXL3>>& gates,
    const std::vector<std::shared_ptr<BC_LinearEXL3>>& ups,
    const std::vector<std::shared_ptr<BC_LinearEXL3>>& downs,
    c10::optional<double> limit)
{
    c10::cuda::CUDAGuard guard(x.device());
    const auto n = gates.size();
    TORCH_CHECK(n > 0 && n <= 6 && ups.size() == n && downs.size() == n
        && x.size(0) == 1, "Invalid expert batch");
    auto projections = gates;
    projections.insert(projections.end(), ups.begin(), ups.end());
    auto gu = grouped_linear(x, projections);
    auto gate = gu.slice(0, 0, n), up = gu.slice(0, n, 2*n);
    if (limit) {
        gate = gate.clamp_max(*limit);
        up = up.clamp(-*limit, *limit);
    }
    auto act = (at::silu(gate) * up).to(at::kHalf).contiguous();
    return grouped_linear(act, downs);
}

at::Tensor grouped_experts(const at::Tensor& x, const at::Tensor& weights,
    const std::vector<int64_t>& positions,
    const std::vector<std::shared_ptr<BC_LinearEXL3>>& gates,
    const std::vector<std::shared_ptr<BC_LinearEXL3>>& ups,
    const std::vector<std::shared_ptr<BC_LinearEXL3>>& downs,
    c10::optional<double> limit, bool fused_reduce)
{
    c10::cuda::CUDAGuard guard(x.device());
    const auto n = gates.size();
    TORCH_CHECK(positions.size() == n && weights.device() == x.device() && weights.dim() == 2
        && weights.size(0) == 1 && weights.size(1) == n, "Invalid weights");
    auto down = grouped_down(x, gates, ups, downs, limit);
    if (fused_reduce) {
        ExpertOrder order{};
        order.count = n;
        for (size_t i = 0; i < n; ++i) {
            TORCH_CHECK(positions[i] >= 0 && positions[i] < n, "Invalid weight index");
            order.positions[i] = positions[i];
        }
        auto result = at::empty_like(x, x.options().dtype(at::kFloat));
        auto w = weights.to(at::kFloat).contiguous();
        launch_expert_sum(down.data_ptr<float>(), w.data_ptr<float>(), result.data_ptr<float>(), x.size(1), order);
        return result;
    }
    auto result = at::zeros_like(x, x.options().dtype(at::kFloat));
    for (size_t i = 0; i < n; ++i) {
        TORCH_CHECK(positions[i] >= 0 && positions[i] < n, "Invalid weight index");
        auto weight = weights.slice(1, positions[i], positions[i] + 1).to(at::kFloat);
        result.add_(down.slice(0, i, i + 1) * weight);
    }
    return result;
}

at::Tensor reduce_split(const at::Tensor& first, const at::Tensor& second,
    const at::Tensor& weights, const std::vector<int64_t>& positions,
    const std::vector<int64_t>& rows)
{
    c10::cuda::CUDAGuard guard(first.device());
    TORCH_CHECK(first.is_cuda() && second.device() == first.device()
        && first.dim() == 2 && second.dim() == 2 && first.size(1) == second.size(1)
        && first.scalar_type() == at::kFloat && second.scalar_type() == at::kFloat
        && first.is_contiguous() && second.is_contiguous(), "Invalid split expert outputs");
    const int n = first.size(1), e = rows.size();
    TORCH_CHECK(e > 0 && e <= 6 && positions.size() == e && first.size(0)+second.size(0) == e
        && weights.device() == first.device() && weights.dim() == 2
        && weights.size(0) == 1 && weights.size(1) == e, "Invalid split routing");
    SplitExpertOrder order{};
    order.count = e;
    for (int i = 0; i < e; ++i) {
        TORCH_CHECK(positions[i] >= 0 && positions[i] < e, "Invalid weight index");
        const auto& source = rows[i] >= 0 ? first : second;
        int row = rows[i] >= 0 ? rows[i] : -1-rows[i];
        TORCH_CHECK(row >= 0 && row < source.size(0), "Invalid split row");
        order.sources[i] = source.data_ptr<float>() + row*n;
        order.positions[i] = positions[i];
    }
    auto result = at::empty({1, n}, first.options());
    auto w = weights.to(at::kFloat).contiguous();
    launch_split_sum(w.data_ptr<float>(), result.data_ptr<float>(), n, order);
    return result;
}

at::Tensor exact_experts(
    const at::Tensor& x,
    const at::Tensor& weights,
    const std::vector<int64_t>& positions,
    const std::vector<std::shared_ptr<BC_LinearEXL3>>& gates,
    const std::vector<std::shared_ptr<BC_LinearEXL3>>& ups,
    const std::vector<std::shared_ptr<BC_LinearEXL3>>& downs,
    c10::optional<double> limit)
{
    c10::cuda::CUDAGuard guard(x.device());
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.size(0) == 1 && x.is_contiguous()
        && x.scalar_type() == at::kHalf, "Expected one contiguous fp16 CUDA row");
    const auto n = gates.size();
    TORCH_CHECK(n > 0 && n == ups.size() && n == downs.size() && n == positions.size(),
        "Expert lists differ");
    TORCH_CHECK(weights.device() == x.device() && weights.dim() == 2
        && weights.size(0) == 1 && weights.size(1) == n, "Invalid routing weights");
    auto result = at::zeros_like(x, x.options().dtype(at::kFloat));
    for (size_t i = 0; i < n; ++i)
    {
        TORCH_CHECK(positions[i] >= 0 && positions[i] < n, "Invalid weight index");
        auto gate = gates[i]->run_alloc(x, gates[i]->svh.numel(), true);
        auto up = ups[i]->run_alloc(x, ups[i]->svh.numel(), true);
        if (limit)
        {
            gate = gate.clamp_max(*limit);
            up = up.clamp(-*limit, *limit);
        }
        auto act = (at::silu(gate) * up).to(at::kHalf).contiguous();
        auto down = downs[i]->run_alloc(act, downs[i]->svh.numel(), true);
        auto weight = weights.slice(1, positions[i], positions[i] + 1).to(at::kFloat);
        result.add_(down * weight);
    }
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("exact_experts", &exact_experts, py::call_guard<py::gil_scoped_release>());
    m.def("grouped_linear", &grouped_linear, py::call_guard<py::gil_scoped_release>());
    m.def("grouped_down", &grouped_down, py::call_guard<py::gil_scoped_release>());
    m.def("reduce_split", &reduce_split, py::call_guard<py::gil_scoped_release>());
    m.def("grouped_experts", &grouped_experts, py::arg("x"), py::arg("weights"), py::arg("positions"),
        py::arg("gates"), py::arg("ups"), py::arg("downs"), py::arg("limit"), py::arg("fused_reduce") = false,
        py::call_guard<py::gil_scoped_release>());
}
