#include "host_planning.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <limits>
#include <utility>
#include <vector>

namespace {

using i32 = int32_t;

torch::Tensor MakeTensor(const std::vector<i32>& values,
                         const std::vector<int64_t>& sizes) {
    if (values.empty()) {
        return torch::empty(sizes, torch::TensorOptions().dtype(torch::kInt32));
    }
    return torch::from_blob(
               const_cast<i32*>(values.data()),
               sizes,
               torch::TensorOptions().dtype(torch::kInt32))
        .clone();
}

int64_t ArgMaxFirst(const std::vector<i32>& values) {
    int64_t best = 0;
    for (int64_t i = 1; i < static_cast<int64_t>(values.size()); ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }
    return best;
}

std::vector<i32> Rebalance(const std::vector<i32>& total,
                           int64_t epn,
                           int64_t cap,
                           int64_t ep_size) {
    const int64_t experts = static_cast<int64_t>(total.size());
    std::vector<i32> alloc(experts * ep_size, 0);
    std::vector<i32> load(ep_size, 0);
    std::vector<i32> spare(ep_size, 0);
    for (int64_t e = 0; e < experts; ++e) {
        const int64_t owner = e / epn;
        alloc[e * ep_size + owner] = total[e];
        load[owner] += total[e];
    }
    for (int64_t d = 0; d < ep_size; ++d) {
        spare[d] = static_cast<i32>(cap) - load[d];
    }

    for (int64_t round = 0; round <= 2 * ep_size; ++round) {
        if (*std::max_element(load.begin(), load.end()) <= cap) {
            break;
        }
        const int64_t home = ArgMaxFirst(load);
        int64_t dest = -1;
        i32 best_spare = -1;
        for (int64_t d = 0; d < ep_size; ++d) {
            const i32 candidate = (d == home) ? -1 : spare[d];
            if (candidate > best_spare) {
                best_spare = candidate;
                dest = d;
            }
        }
        if (dest < 0) {
            break;
        }
        i32 move = std::min<i32>(load[home] - static_cast<i32>(cap),
                                 spare[dest]);
        if (move <= 0) {
            break;
        }
        while (move > 0) {
            int64_t local = -1;
            i32 held = -1;
            for (int64_t i = 0; i < epn; ++i) {
                const i32 candidate =
                    alloc[(home * epn + i) * ep_size + home];
                if (candidate > held) {
                    held = candidate;
                    local = i;
                }
            }
            if (local < 0) {
                break;
            }
            const i32 take = std::min(held, move);
            if (take <= 0) {
                break;
            }
            const int64_t expert = home * epn + local;
            alloc[expert * ep_size + dest] += take;
            alloc[expert * ep_size + home] -= take;
            load[home] -= take;
            load[dest] += take;
            spare[home] += take;
            spare[dest] -= take;
            move -= take;
        }
    }
    return alloc;
}

}  // namespace

HostPlanResult BuildHostPlan(
    const torch::Tensor& counts_all,
    const torch::Tensor& local_counts,
    const torch::Tensor& route_ids,
    int64_t rank,
    int64_t epn,
    int64_t nvs) {
    TORCH_CHECK(counts_all.is_cpu() && local_counts.is_cpu() &&
                    route_ids.is_cpu(),
                "host plan inputs must be CPU tensors");
    TORCH_CHECK(counts_all.scalar_type() == torch::kInt32 &&
                    local_counts.scalar_type() == torch::kInt32 &&
                    route_ids.scalar_type() == torch::kInt32,
                "host plan inputs must be int32 tensors");
    TORCH_CHECK(counts_all.is_contiguous() && local_counts.is_contiguous() &&
                    route_ids.is_contiguous(),
                "host plan inputs must be contiguous");
    TORCH_CHECK(counts_all.dim() == 2 && local_counts.dim() == 1 &&
                    route_ids.dim() == 1,
                "expected counts_all [R, E], local_counts [E], route_ids [N]");

    const int64_t world = counts_all.size(0);
    const int64_t experts = counts_all.size(1);
    const int64_t rows = route_ids.numel();
    const int64_t cap = nvs;
    TORCH_CHECK(world >= 1 && experts >= 1, "R and E must be positive");
    TORCH_CHECK(epn >= 1 && experts == world * epn,
                "expert count must equal R * epn");
    TORCH_CHECK(cap >= 0 && rows == cap,
                "route_ids must hold exactly NvS entries");
    TORCH_CHECK(rank >= 0 && rank < world, "rank out of range");
    TORCH_CHECK(local_counts.numel() == experts,
                "local_counts must have E entries");

    const i32* counts_ptr = counts_all.data_ptr<i32>();
    const i32* local_ptr = local_counts.data_ptr<i32>();
    const i32* route_ptr = route_ids.data_ptr<i32>();

    TORCH_CHECK(cap <= std::numeric_limits<i32>::max() / world,
                "R * NvS exceeds int32 destination range");
    std::vector<int64_t> histogram(experts, 0);
    for (int64_t i = 0; i < rows; ++i) {
        TORCH_CHECK(route_ptr[i] >= 0 && route_ptr[i] < experts,
                    "route_ids must be in [0, E)");
        ++histogram[route_ptr[i]];
    }
    for (int64_t e = 0; e < experts; ++e) {
        TORCH_CHECK(local_ptr[e] >= 0, "local_counts must be nonnegative");
        TORCH_CHECK(local_ptr[e] == counts_ptr[rank * experts + e],
                    "local_counts must match counts_all[rank]");
        TORCH_CHECK(histogram[e] == local_ptr[e],
                    "route_ids histogram must match local_counts");
        for (int64_t r = 0; r < world; ++r) {
            TORCH_CHECK(counts_ptr[r * experts + e] >= 0,
                        "counts_all must be nonnegative");
        }
    }

    std::vector<i32> total(experts, 0);
    int64_t local_sum = 0;
    for (int64_t e = 0; e < experts; ++e) {
        local_sum += local_ptr[e];
        for (int64_t r = 0; r < world; ++r) {
            total[e] += counts_ptr[r * experts + e];
        }
    }
    TORCH_CHECK(local_sum == cap,
                "local_counts sum to ", local_sum, " rows, expected ", cap);
    if (cap > 0) {
        i32 min_id = route_ptr[0];
        i32 max_id = route_ptr[0];
        for (int64_t i = 1; i < rows; ++i) {
            min_id = std::min(min_id, route_ptr[i]);
            max_id = std::max(max_id, route_ptr[i]);
        }
        TORCH_CHECK(min_id >= 0 && max_id < experts,
                    "route_ids must be in [0, E)");
    }
    for (int64_t r = 0; r < world; ++r) {
        int64_t sum = 0;
        for (int64_t e = 0; e < experts; ++e) {
            sum += counts_ptr[r * experts + e];
        }
        TORCH_CHECK(sum == cap,
                    "every rank must present exactly NvS rows");
    }

    const int64_t slots = epn;
    const int64_t groups = experts + slots;
    std::vector<i32> alloc = Rebalance(total, epn, cap, world);

    // Every expert whose rows land on a non-owner rank.
    std::vector<i32> needed_remote;
    for (int64_t e = 0; e < experts; ++e) {
        const int64_t owner = e / epn;
        bool needed = false;
        for (int64_t d = 0; d < world; ++d) {
            if (d != owner && alloc[e * world + d] > 0) {
                needed = true;
                break;
            }
        }
        if (needed) {
            needed_remote.push_back(static_cast<i32>(e));
        }
    }

    // Prefetch picks: top-B remote experts per destination, ordered by
    // (rows, expert id) descending, matching the Python planner.
    std::vector<i32> copies(world * slots, -1);
    std::vector<i32> remote_stats(world * 2, 0);
    std::vector<uint8_t> prefetch_mask(world * experts, 0);
    for (int64_t d = 0; d < world; ++d) {
        std::vector<std::pair<int64_t, int64_t>> eligible;
        for (int64_t e = 0; e < experts; ++e) {
            if (e / epn == d || alloc[e * world + d] <= 0) {
                continue;
            }
            const int64_t key =
                static_cast<int64_t>(alloc[e * world + d]) * (experts + 1) + e;
            eligible.emplace_back(key, e);
        }
        remote_stats[d * 2] = static_cast<i32>(eligible.size());
        std::sort(eligible.begin(), eligible.end(),
                  [](const auto& lhs, const auto& rhs) {
                      return lhs.first > rhs.first;
                  });
        const int64_t take = std::min<int64_t>(slots, eligible.size());
        for (int64_t j = 0; j < take; ++j) {
            const int64_t expert = eligible[j].second;
            copies[d * slots + j] = static_cast<i32>(expert);
            prefetch_mask[d * experts + expert] = 1;
            remote_stats[(expert / epn) * 2 + 1] += 1;
        }
    }

    std::vector<i32> group_counts(world * groups, 0);
    std::vector<i32> all_counts(world * groups, 0);
    std::vector<i32> group_start(world * groups, 0);
    std::vector<i32> cu_seqlens(world * groups, 0);
    std::vector<i32> expert_off(world * experts, 0);
    for (int64_t d = 0; d < world; ++d) {
        for (int64_t e = 0; e < experts; ++e) {
            if (alloc[e * world + d] > 0 && !prefetch_mask[d * experts + e]) {
                group_counts[d * groups + e] = alloc[e * world + d];
            }
        }
        for (int64_t j = 0; j < slots; ++j) {
            const int64_t expert = copies[d * slots + j];
            if (expert >= 0) {
                group_counts[d * groups + experts + j] =
                    alloc[expert * world + d];
            }
        }
        i32 running = 0;
        for (int64_t g = 0; g < groups; ++g) {
            group_start[d * groups + g] = running;
            running += group_counts[d * groups + g];
            all_counts[d * groups + g] = running;
            cu_seqlens[d * groups + g] = running;
        }
        for (int64_t e = 0; e < experts; ++e) {
            if (group_counts[d * groups + e] > 0) {
                expert_off[d * experts + e] = group_start[d * groups + e];
            }
        }
        for (int64_t j = 0; j < slots; ++j) {
            const int64_t expert = copies[d * slots + j];
            if (expert >= 0) {
                expert_off[d * experts + expert] =
                    group_start[d * groups + experts + j];
            }
        }
    }

    // Prefix counts over source ranks: rows of expert e on ranks < rank.
    std::vector<i32> prefix(world * experts, 0);
    for (int64_t r = 1; r < world; ++r) {
        for (int64_t e = 0; e < experts; ++e) {
            prefix[r * experts + e] =
                prefix[(r - 1) * experts + e] + counts_ptr[(r - 1) * experts + e];
        }
    }
    std::vector<i32> alloc_cumsum(experts * world, 0);
    for (int64_t e = 0; e < experts; ++e) {
        i32 running = 0;
        for (int64_t d = 0; d < world; ++d) {
            running += alloc[e * world + d];
            alloc_cumsum[e * world + d] = running;
        }
    }

    std::vector<i32> dst(rows, 0);
    std::vector<i32> seen(experts, 0);
    for (int64_t i = 0; i < rows; ++i) {
        const int64_t expert = route_ptr[i];
        const i32 local_index = seen[expert]++;
        const i32 global_index =
            prefix[rank * experts + expert] + local_index;
        int64_t dest = 0;
        while (dest < world &&
               alloc_cumsum[expert * world + dest] <= global_index) {
            ++dest;
        }
        TORCH_CHECK(dest < world, "no destination rank for a routed row");
        const i32 previous =
            dest == 0 ? 0 : alloc_cumsum[expert * world + dest - 1];
        dst[i] = static_cast<i32>(dest) * static_cast<i32>(cap) +
                 expert_off[dest * experts + expert] +
                 (global_index - previous);
    }

    std::vector<i32> alloc_re(world * experts, 0);
    for (int64_t d = 0; d < world; ++d) {
        for (int64_t e = 0; e < experts; ++e) {
            alloc_re[d * experts + e] = alloc[e * world + d];
        }
    }
    std::vector<i32> cu_rank(groups, 0);
    for (int64_t g = 0; g < groups; ++g) {
        cu_rank[g] = cu_seqlens[rank * groups + g];
    }
    std::vector<i32> stats_rank(2, 0);
    stats_rank[0] = remote_stats[rank * 2];
    stats_rank[1] = remote_stats[rank * 2 + 1];
    std::vector<i32> zero_fill(groups * 2, 0);

    const int64_t lo = rank * epn;
    HostPlanResult result;
    result.dst = MakeTensor(dst, {rows});
    result.cu_seqlens = MakeTensor(cu_rank, {groups});
    result.experts_to_copy = MakeTensor(copies, {world, slots});
    result.remote_stats = MakeTensor(stats_rank, {2});
    result.zero_fill_ranges = MakeTensor(zero_fill, {groups, 2});
    result.alloc = MakeTensor(alloc_re, {world, experts});
    result.needed_remote = MakeTensor(
        needed_remote, {static_cast<int64_t>(needed_remote.size())});
    result.span_begin = group_start[rank * groups + lo];
    result.span_end = cu_seqlens[rank * groups + lo + epn - 1];
    return result;
}
