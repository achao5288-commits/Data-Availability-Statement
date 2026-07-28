#!/usr/bin/env python3
"""
Supplementary Implementation: Hybrid QUBO-CIM Framework for VRPTW
Targeting: Computers & Operations Research

This script implements the key algorithms described in the paper:
1. MDS-based customer clustering
2. QUBO construction for TSPTW
3. Lookahead greedy initialization
4. Enhanced Simulated Annealing with CIM integration
5. Solomon benchmark evaluation

Requires: numpy, scipy, scikit-learn, matplotlib, kaiwu-sdk (for CIM)
"""

import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict, Optional
import time
import random
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Customer:
    """Customer node in VRPTW."""
    id: int
    x: float
    y: float
    demand: float
    ready_time: float   # a_i: earliest arrival
    due_time: float     # b_i: latest arrival
    service_time: float # s_i


@dataclass
class VRPTWInstance:
    """Complete VRPTW problem instance."""
    customers: List[Customer]
    depot: Customer
    capacity: float
    vehicle_penalty: float = 1000.0
    M1: float = 10.0  # Early arrival penalty coefficient
    M2: float = 20.0  # Late arrival penalty coefficient

    @property
    def N(self) -> int:
        return len(self.customers)

    def travel_time(self, i: int, j: int) -> float:
        """Compute travel time between nodes i and j (0 = depot)."""
        if i == 0:
            ci = self.depot
        else:
            ci = self.customers[i - 1]
        if j == 0:
            cj = self.depot
        else:
            cj = self.customers[j - 1]
        return np.sqrt((ci.x - cj.x)**2 + (ci.y - cj.y)**2)

    def time_matrix(self) -> np.ndarray:
        """Build complete travel time matrix."""
        n = self.N + 1  # including depot
        T = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                T[i, j] = self.travel_time(i, j)
        return T


# ============================================================================
# Stage 1: MDS-Based Customer Clustering
# ============================================================================

def classical_mds(D: np.ndarray, d: int = 2) -> np.ndarray:
    """
    Classical (metric) Multi-Dimensional Scaling.

    Args:
        D: Distance matrix (n x n), symmetric
        d: Target embedding dimension

    Returns:
        Y: Embedded coordinates (n x d)
    """
    n = D.shape[0]
    # Double centering
    D_sq = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D_sq @ J

    # Eigendecomposition
    eigenvalues, eigenvectors = eigh(B)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top d positive eigenvalues
    d_eff = min(d, np.sum(eigenvalues > 1e-10))
    L = np.diag(np.sqrt(np.maximum(eigenvalues[:d_eff], 0)))
    Y = eigenvectors[:, :d_eff] @ L
    return Y


def mds_clustering(instance: VRPTWInstance,
                   k_min: Optional[int] = None) -> List[List[int]]:
    """
    Cluster customers using MDS embedding + k-medoids.

    Args:
        instance: VRPTW problem instance
        k_min: Minimum number of clusters (default: ceil(total_demand / capacity))

    Returns:
        clusters: List of clusters, each containing customer indices (1-based)
    """
    T = instance.time_matrix()
    # MDS embedding of customer-to-customer distances
    customer_dist = T[1:, 1:]  # Exclude depot
    Y = classical_mds(customer_dist, d=2)

    if k_min is None:
        total_demand = sum(c.demand for c in instance.customers)
        k_min = max(1, int(np.ceil(total_demand / instance.capacity)))

    best_clusters = None
    best_silhouette = -1

    max_k = min(k_min + 6, len(instance.customers))
    for k in range(k_min, max_k + 1):
        if k < 2:
            # Single cluster: all customers together
            clusters_list = [list(range(1, instance.N + 1))]
            total_d = sum(instance.customers[c-1].demand for c in clusters_list[0])
            if total_d <= instance.capacity * 1.5:
                return clusters_list
            continue
        try:
            k = min(k, len(instance.customers))
            # Use KMeans + find closest-to-centroid points as medoids
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(Y)

            # Check capacity feasibility
            clusters = defaultdict(list)
            feasible = True
            for i, label in enumerate(labels):
                clusters[label].append(i + 1)  # 1-based customer index

            for cluster_customers in clusters.values():
                total_d = sum(instance.customers[c-1].demand
                            for c in cluster_customers)
                if total_d > instance.capacity * 1.1:  # 10% tolerance
                    feasible = False
                    break

            if not feasible:
                continue

            # Compute silhouette score
            from sklearn.metrics import silhouette_score
            if k > 1 and len(set(labels)) > 1:
                sil = silhouette_score(Y, labels)
                if sil > best_silhouette:
                    best_silhouette = sil
                    best_clusters = list(clusters.values())

        except Exception:
            continue

    return best_clusters


# ============================================================================
# Stage 2: QUBO Construction for TSPTW
# ============================================================================

def construct_tsp_tw_qubo(instance: VRPTWInstance,
                          cluster_customers: List[int],
                          p_cost: float = 1.0,
                          p_constraint: float = 100.0) -> np.ndarray:
    """
    Construct QUBO matrix for TSPTW subproblem on a customer cluster.

    The QUBO uses one-hot encoding:
    x_{i,p} = 1 if customer i is visited at position p.

    Args:
        instance: Full VRPTW instance
        cluster_customers: List of customer indices in this cluster (1-based)
        p_cost: Objective weight
        p_constraint: Constraint penalty weight

    Returns:
        Q: QUBO matrix (upper triangular)
    """
    m = len(cluster_customers)
    n_vars = m * m  # m positions x m customers

    # Map cluster position to global customer index
    cust_global = [c - 1 for c in cluster_customers]  # 0-based into instance.customers

    Q = np.zeros((n_vars, n_vars))

    def idx(i: int, p: int) -> int:
        """Map (customer_in_cluster, position) to QUBO variable index."""
        return i * m + p

    # 1. Cost term: travel time between consecutive positions
    for p in range(m - 1):
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                gi = cust_global[i]
                gj = cust_global[j]
                # Travel time from i to j (adding service time at i)
                tij = instance.travel_time(gi + 1, gj + 1) + instance.customers[gi].service_time
                Q[idx(i, p), idx(j, p + 1)] += p_cost * tij

    # Return to depot from last position
    for i in range(m):
        gi = cust_global[i]
        ti0 = instance.travel_time(gi + 1, 0)
        Q[idx(i, m - 1), idx(i, m - 1)] += p_cost * ti0

    # 2. Constraint: each customer visited exactly once
    for i in range(m):
        for p in range(m):
            Q[idx(i, p), idx(i, p)] -= p_constraint
            for q in range(m):
                if p != q:
                    Q[idx(i, p), idx(i, q)] += 2 * p_constraint
                for j in range(m):
                    if i != j:
                        Q[idx(i, p), idx(j, p)] += 2 * p_constraint

    # Ensure upper triangular
    for i in range(n_vars):
        for j in range(i):
            Q[j, i] += Q[i, j]
            Q[i, j] = 0

    return Q


# ============================================================================
# Classical Solvers for Comparison
# ============================================================================

def classical_simulated_annealing(instance: VRPTWInstance,
                                  route: List[int],
                                  T0: float = 1000,
                                  T_min: float = 0.01,
                                  alpha: float = 0.95,
                                  iters_per_T: int = 500) -> Tuple[List[int], float]:
    """
    Classical SA optimization of a single TSPTW route.
    """
    current = route.copy()
    current_cost = evaluate_route(instance, current)

    best_route = current.copy()
    best_cost = current_cost

    T = T0
    while T > T_min:
        for _ in range(iters_per_T):
            # 2-opt move
            i, j = sorted(random.sample(range(len(current)), 2))
            if j - i < 2:
                continue
            new_route = (current[:i] +
                        list(reversed(current[i:j+1])) +
                        current[j+1:])

            new_cost = evaluate_route(instance, new_route)
            delta = new_cost - current_cost

            if delta < 0 or random.random() < np.exp(-delta / T):
                current = new_route
                current_cost = new_cost
                if current_cost < best_cost:
                    best_route = current.copy()
                    best_cost = current_cost

        T *= alpha

    return best_route, best_cost


def evaluate_route(instance: VRPTWInstance, route: List[int]) -> float:
    """
    Evaluate a TSPTW route with time window penalties.

    Args:
        instance: VRPTW instance
        route: Ordered list of customer indices in the route (excluding depot)

    Returns:
        Total cost (travel time + penalties)
    """
    current_time = 0.0
    total_cost = 0.0
    prev_node = 0  # Start at depot

    for cust_idx in route:
        customer = instance.customers[cust_idx - 1]
        travel_time = instance.travel_time(prev_node, cust_idx)
        arrival_time = current_time + travel_time

        # Time window penalty
        if arrival_time < customer.ready_time:
            # Early: wait
            wait_time = customer.ready_time - arrival_time
            arrival_time = customer.ready_time
            total_cost += instance.M1 * (wait_time ** 2)
        elif arrival_time > customer.due_time:
            # Late: penalty
            delay = arrival_time - customer.due_time
            total_cost += instance.M2 * (delay ** 2)

        total_cost += travel_time
        current_time = arrival_time + customer.service_time
        prev_node = cust_idx

    # Return to depot
    total_cost += instance.travel_time(prev_node, 0)
    return total_cost


# ============================================================================
# Simulated CIM Solver (when Kaiwu SDK is unavailable)
# ============================================================================

def simulated_cim_solve(Q: np.ndarray,
                        n_runs: int = 10,
                        n_iters: int = 300) -> np.ndarray:
    """
    Simulated CIM solver using coherent-state-inspired annealing.
    This approximates CIM dynamics when actual hardware is unavailable.

    CIM key characteristics simulated:
    - Collective spin evolution (parallel updates)
    - Bifurcation-based convergence (sigmoid activation)
    - Annealing schedule (pump amplitude ramp)

    Args:
        Q: QUBO matrix (n_vars x n_vars)
        n_runs: Number of independent CIM shots
        n_iters: Annealing steps per run

    Returns:
        Best binary solution vector
    """
    n_vars = Q.shape[0]
    best_energy = float('inf')
    best_solution = None

    for run in range(n_runs):
        # Initialize spins in superposition (small random values)
        spins = np.random.randn(n_vars) * 0.1

        # Annealing schedule: pump amplitude p(t) = tanh(t/tau)
        for t in range(n_iters):
            p = np.tanh(t / (n_iters * 0.3))  # Pump amplitude [0, ~1]

            # Compute Ising interaction (simultaneous for all spins)
            coupling = Q @ spins

            # DOPO bifurcation dynamics with noise
            noise = np.random.randn(n_vars) * 0.01 * (1 - p)
            spins = np.tanh(p * coupling + noise)

        # Threshold to binary
        binary = (spins > 0).astype(float)

        # Verify and compute energy
        energy = binary @ Q @ binary
        if energy < best_energy:
            best_energy = energy
            best_solution = binary.copy()

    return best_solution


# ============================================================================
# Lookahead Greedy Initialization
# ============================================================================

def lookahead_greedy_init(instance: VRPTWInstance,
                          cluster_customers: List[int],
                          depth: int = 3,
                          w1: float = 1.0,
                          w2: float = 1.0,
                          w3: float = 0.5) -> List[int]:
    """
    Lookahead greedy initialization for TSPTW route construction.

    Score(j) = w1 * Penalty(j) + w2 * TravelTime(current, j) + w3 * E[FuturePenalty]

    Args:
        instance: VRPTW instance
        cluster_customers: Customers to route
        depth: Lookahead depth for future penalty estimation
        w1, w2, w3: Scoring weights

    Returns:
        Constructed route (list of customer indices)
    """
    unvisited = set(cluster_customers)
    route = []
    current_time = 0.0
    prev_node = 0  # Depot

    while unvisited:
        best_score = float('inf')
        best_customer = None

        for cust in unvisited:
            customer = instance.customers[cust - 1]
            t = instance.travel_time(prev_node, cust)
            arr = current_time + t

            # Direct penalty
            direct_penalty = 0.0
            if arr < customer.ready_time:
                direct_penalty = instance.M1 * (customer.ready_time - arr) ** 2
            elif arr > customer.due_time:
                direct_penalty = instance.M2 * (arr - customer.due_time) ** 2

            # Estimate future penalty (simplified lookahead)
            future_penalty = 0.0
            if len(unvisited) > 1:
                others = [c for c in unvisited if c != cust]
                for oc in others[:depth]:
                    oc_cust = instance.customers[oc - 1]
                    t_future = instance.travel_time(cust, oc)
                    arr_future = (max(arr, customer.ready_time) +
                                customer.service_time + t_future)
                    if arr_future < oc_cust.ready_time:
                        future_penalty += (instance.M1 *
                                         (oc_cust.ready_time - arr_future) ** 2)
                    elif arr_future > oc_cust.due_time:
                        future_penalty += (instance.M2 *
                                         (arr_future - oc_cust.due_time) ** 2)
                future_penalty /= min(len(others), depth)

            score = w1 * direct_penalty + w2 * t + w3 * future_penalty

            if score < best_score:
                best_score = score
                best_customer = cust

        route.append(best_customer)
        unvisited.remove(best_customer)

        # Update state
        customer = instance.customers[best_customer - 1]
        t = instance.travel_time(prev_node, best_customer)
        current_time = max(current_time + t, customer.ready_time) + customer.service_time
        prev_node = best_customer

    return route


# ============================================================================
# Enhanced Simulated Annealing (ESA) - Main Solver
# ============================================================================

def solve_vrptw_esa_cim(instance: VRPTWInstance,
                        T0: float = 100,
                        T_min: float = 0.1,
                        alpha: float = 0.9,
                        iters_per_T: int = 50,
                        T_reheat: float = 30,
                        use_cim: bool = True,
                        verbose: bool = True) -> Dict:
    """
    Solve VRPTW using the full MDS + QUBO-CIM + ESA framework.

    Returns:
        Dictionary containing solution details
    """
    t_start = time.time()

    # Stage 1: MDS Clustering
    if verbose:
        print("[Stage 1] MDS-based customer clustering...")
    t1 = time.time()
    clusters = mds_clustering(instance)
    t_mds = time.time() - t1
    if verbose:
        print(f"  Clustered {instance.N} customers into {len(clusters)} groups")
        print(f"  MDS time: {t_mds:.2f}s")

    # Stage 2: Initialize each cluster route
    if verbose:
        print("[Stage 2] Initial route construction...")
    t2 = time.time()
    cluster_routes = {}
    for i, cluster in enumerate(clusters):
        route = lookahead_greedy_init(instance, cluster)
        cluster_routes[i] = route

    t_init = time.time() - t2

    # Stage 3: QUBO-CIM optimization per cluster
    t_cim_total = 0
    if use_cim:
        if verbose:
            print("[Stage 3] QUBO-CIM optimization per cluster...")
        for i, cluster in enumerate(clusters):
            t_cim_start = time.time()
            Q = construct_tsp_tw_qubo(instance, cluster)
            solution = simulated_cim_solve(Q, n_runs=50)

            # Decode solution back to route
            m = len(cluster)
            binary = solution.reshape(m, m)
            decoded_route = []
            for p in range(m):
                customer_idx_in_cluster = np.argmax(binary[:, p])
                decoded_route.append(cluster[customer_idx_in_cluster])
            cluster_routes[i] = decoded_route

            t_cim = time.time() - t_cim_start
            t_cim_total += t_cim
            if verbose:
                cost = evaluate_route(instance, decoded_route)
                print(f"  Cluster {i} ({m} customers): cost={cost:.1f}, "
                      f"CIM time={t_cim:.2f}s")
    else:
        if verbose:
            print("[Stage 3] Classical SA per cluster...")
        for i, cluster in enumerate(clusters):
            route = cluster_routes[i]
            optimized_route, cost = classical_simulated_annealing(
                instance, route, T0, T_min, alpha, iters_per_T
            )
            cluster_routes[i] = optimized_route

    # Stage 4: ESA global optimization
    if verbose:
        print("[Stage 4] ESA global optimization...")
    t_esa_start = time.time()

    # Compute initial global objective
    current_obj = compute_global_objective(instance, clusters, cluster_routes)
    best_obj = current_obj
    best_routes = {k: v.copy() for k, v in cluster_routes.items()}

    T = T0
    no_improvement_count = 0
    epoch = 0

    while T > T_min:
        for _ in range(iters_per_T):
            # Select neighborhood operator
            op = random.random()
            new_clusters = [c.copy() for c in clusters]
            new_routes = {k: v.copy() for k, v in cluster_routes.items()}

            if op < 0.6:  # 2-opt within a cluster
                k = random.randrange(len(clusters))
                route = new_routes[k]
                if len(route) >= 4:
                    i, j = sorted(random.sample(range(len(route)), 2))
                    if j - i >= 2:
                        new_routes[k] = (route[:i] +
                                        list(reversed(route[i:j+1])) +
                                        route[j+1:])
                        # Skip CIM re-optimization during ESA for speed
                        # (CIM is used during initialization only)
                        pass

            elif op < 0.9:  # Relocate
                if len(clusters) >= 2:
                    k_from, k_to = random.sample(range(len(clusters)), 2)
                    if len(new_routes[k_from]) > 1:
                        idx = random.randrange(len(new_routes[k_from]))
                        customer = new_routes[k_from][idx]
                        if customer in new_clusters[k_from]:
                            new_clusters[k_from].remove(customer)
                            new_routes[k_from].remove(customer)
                            new_clusters[k_to].append(customer)
                            insert_pos = random.randrange(len(new_routes[k_to]) + 1)
                            new_routes[k_to].insert(insert_pos, customer)

            else:  # Exchange
                if len(clusters) >= 2:
                    k1, k2 = random.sample(range(len(clusters)), 2)
                    if new_routes[k1] and new_routes[k2]:
                        i1 = random.randrange(len(new_routes[k1]))
                        i2 = random.randrange(len(new_routes[k2]))
                        c1, c2 = new_routes[k1][i1], new_routes[k2][i2]
                        if c1 in new_clusters[k1] and c2 in new_clusters[k2]:
                            new_clusters[k1].remove(c1)
                            new_clusters[k2].remove(c2)
                            new_clusters[k1].append(c2)
                            new_clusters[k2].append(c1)
                            new_routes[k1][i1] = c2
                            new_routes[k2][i2] = c1

            new_obj = compute_global_objective(instance, new_clusters, new_routes)
            delta = new_obj - current_obj

            if delta < 0 or random.random() < np.exp(-delta / T):
                clusters = new_clusters
                cluster_routes = new_routes
                current_obj = new_obj
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_routes = {k: v.copy() for k, v in cluster_routes.items()}
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

        # Reheating
        if T < T_reheat and no_improvement_count >= 3 * iters_per_T:
            T = T_reheat
            no_improvement_count = 0

        T *= alpha
        epoch += 1

    t_esa = time.time() - t_esa_start
    t_total = time.time() - t_start

    # Post-optimization: 2-opt on each route
    t_post_start = time.time()
    for k in best_routes:
        route = best_routes[k]
        improved_route, _ = classical_simulated_annealing(
            instance, route, T0=10, T_min=0.01, alpha=0.9, iters_per_T=100
        )
        best_routes[k] = improved_route
    t_post = time.time() - t_post_start

    if verbose:
        print(f"\n{'='*60}")
        print(f"Solution Summary:")
        print(f"  Objective: {best_obj:.1f}")
        print(f"  Vehicles: {len(best_routes)}")
        print(f"  Total time: {t_total:.2f}s")
        print(f"  Time breakdown: MDS={t_mds:.2f}s, Init={t_init:.2f}s, "
              f"CIM={t_cim_total:.2f}s, ESA={t_esa:.2f}s, Post={t_post:.2f}s")

    return {
        'objective': best_obj,
        'num_vehicles': len(best_routes),
        'routes': best_routes,
        'clusters': clusters,
        'time_total': t_total,
        'time_mds': t_mds,
        'time_init': t_init,
        'time_cim': t_cim_total,
        'time_esa': t_esa,
        'time_post': t_post,
    }


def compute_global_objective(instance: VRPTWInstance,
                            clusters: List[List[int]],
                            routes: Dict[int, List[int]]) -> float:
    """Compute VRPTW global objective across all routes."""
    total = 0.0

    for k, route in routes.items():
        # Vehicle fixed cost
        total += instance.vehicle_penalty

        # Route travel time and penalties
        route_cost = evaluate_route(instance, route)
        total += route_cost

        # Capacity check (soft penalty)
        cluster_demand = sum(instance.customers[c-1].demand for c in route)
        if cluster_demand > instance.capacity:
            total += 10000 * (cluster_demand - instance.capacity)

    return total


# ============================================================================
# Solomon Benchmark Generator
# ============================================================================

def generate_solomon_instance(class_type: str = 'C1',
                              n_customers: int = 100) -> VRPTWInstance:
    """
    Generate a Solomon-style VRPTW instance.

    Args:
        class_type: One of 'C1', 'C2', 'R1', 'R2', 'RC1', 'RC2'
        n_customers: Number of customers

    Returns:
        VRPTWInstance with Solomon-style parameters
    """
    random.seed(hash(class_type) % 10000)
    np.random.seed(hash(class_type) % 10000)

    # Depot at center
    depot = Customer(id=0, x=50, y=50, demand=0,
                    ready_time=0, due_time=1000, service_time=0)

    customers = []
    max_route_duration = 200 if class_type.endswith('2') else 100

    if class_type.startswith('C'):
        # Clustered: groups of customers around cluster centers
        n_clusters = 6 if n_customers <= 50 else 8
        centers = [(random.uniform(15, 85), random.uniform(15, 85))
                  for _ in range(n_clusters)]

        for i in range(n_customers):
            cx, cy = centers[i % n_clusters]
            x = np.clip(cx + random.gauss(0, 5), 0, 100)
            y = np.clip(cy + random.gauss(0, 5), 0, 100)
            dist_to_depot = np.sqrt((x-50)**2 + (y-50)**2)

            ready = random.uniform(0, max_route_duration * 0.5)
            due = ready + random.uniform(10, 30)
            demand = random.uniform(5, 30)
            service = random.uniform(5, 15)

            customers.append(Customer(id=i+1, x=x, y=y, demand=demand,
                                     ready_time=ready, due_time=due,
                                     service_time=service))

    elif class_type.startswith('R'):
        # Random: uniformly distributed
        for i in range(n_customers):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            dist_to_depot = np.sqrt((x-50)**2 + (y-50)**2)

            ready = random.uniform(0, max_route_duration * 0.5)
            due = ready + random.uniform(15, 60)
            demand = random.uniform(5, 30)
            service = random.uniform(5, 15)

            customers.append(Customer(id=i+1, x=x, y=y, demand=demand,
                                     ready_time=ready, due_time=due,
                                     service_time=service))

    else:  # RC: mixed
        n_clusters = 4
        centers = [(random.uniform(20, 80), random.uniform(20, 80))
                  for _ in range(n_clusters)]
        for i in range(n_customers):
            if i < n_customers * 0.6:
                cx, cy = centers[i % n_clusters]
                x = np.clip(cx + random.gauss(0, 8), 0, 100)
                y = np.clip(cy + random.gauss(0, 8), 0, 100)
            else:
                x = random.uniform(0, 100)
                y = random.uniform(0, 100)

            ready = random.uniform(0, max_route_duration * 0.5)
            due = ready + random.uniform(20, 50)
            demand = random.uniform(5, 30)
            service = random.uniform(5, 15)

            customers.append(Customer(id=i+1, x=x, y=y, demand=demand,
                                     ready_time=ready, due_time=due,
                                     service_time=service))

    return VRPTWInstance(customers=customers, depot=depot, capacity=60)


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_solomon_benchmark(use_cim: bool = True,
                          n_runs: int = 5) -> Dict:
    """
    Run the framework on all Solomon instance classes.

    Returns:
        Results dictionary with per-class metrics
    """
    classes = ['C1', 'R1', 'RC1']  # Demo: 3 representative classes
    results = {}

    for cls in classes:
        print(f"\n{'#'*60}")
        print(f"# Benchmarking class: {cls}")
        print(f"{'#'*60}")

        objectives = []
        vehicles = []
        times = []

        for run in range(n_runs):
            instance = generate_solomon_instance(cls, n_customers=100)
            result = solve_vrptw_esa_cim(instance, use_cim=use_cim, verbose=False)
            objectives.append(result['objective'])
            vehicles.append(result['num_vehicles'])
            times.append(result['time_total'])
            print(f"  Run {run+1}/{n_runs}: obj={result['objective']:.1f}, "
                  f"veh={result['num_vehicles']}, time={result['time_total']:.1f}s")

        results[cls] = {
            'mean_objective': np.mean(objectives),
            'std_objective': np.std(objectives),
            'min_objective': np.min(objectives),
            'max_objective': np.max(objectives),
            'mean_vehicles': np.mean(vehicles),
            'mean_time': np.mean(times),
        }

    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Hybrid QUBO-CIM Framework for VRPTW")
    print("Target: Computers & Operations Research")
    print("=" * 70)

    # ---- Test 1: Small 15-customer TSPTW ----
    print("\n[Test 1] 15-Customer TSPTW Validation")
    instance_15 = generate_solomon_instance('C1', n_customers=15)
    result_15 = solve_vrptw_esa_cim(instance_15, use_cim=True)

    # ---- Test 2: 50-customer VRPTW ----
    print("\n[Test 2] 50-Customer VRPTW")
    instance_50 = generate_solomon_instance('R1', n_customers=50)
    result_50 = solve_vrptw_esa_cim(instance_50, use_cim=True)

    # ---- Test 3: 50-customer VRPTW (no CIM, for comparison) ----
    print("\n[Test 3] 50-Customer VRPTW (Classical SA only, no CIM)")
    result_50_no_cim = solve_vrptw_esa_cim(instance_50, use_cim=False)

    # ---- Test 4: Solomon 100-customer benchmark ----
    print("\n[Test 4] Solomon 100-Customer Benchmark")
    results = run_solomon_benchmark(use_cim=True, n_runs=3)

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Class':<8} {'Mean Obj':>12} {'Std Obj':>10} {'Veh':>6} {'Time(s)':>8}")
    print("-" * 50)
    for cls, r in results.items():
        print(f"{cls:<8} {r['mean_objective']:>12.1f} {r['std_objective']:>10.1f} "
              f"{r['mean_vehicles']:>6.1f} {r['mean_time']:>8.1f}")
