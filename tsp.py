import numpy as np
from python_tsp.heuristics import solve_tsp_simulated_annealing

"""
What changes should I do to apply this to distance matrixes with circular pattern

TODO
- Total dist to use distance matrix instead of just indices
- Do not use error as a break point as it is impossible to generate for my cases
- adjacent routine sometimes doesn't swap values
- Add minimize and maximize functions
"""

def total_dist(route, dist_mat, linear = False):
    d = 0.0
    n = len(route)

    mod = 0
    if linear:
        mod = 1

    for i in range(n - mod):
        d += dist_mat[route[i], route[(i + 1) % n]]

    return d

def adjacent(route):
  n = len(route)
  result = np.copy(route)
  i = np.random.randint(n)
  j = np.random.randint(n)
  tmp = result[i]
  result[i] = result[j]
  result[j] = tmp

  return result

def start_temp(dist_mat, route, linear = False):
    """
    Choose starting temp depeding on:
    Dréo, Johann, et al. Metaheuristics for hard optimization: methods and case studies. Springer Science & Business Media, 2006.
    """
    temp_list = np.zeros([100])
    route_dist = total_dist(route, dist_mat, linear)

    for i in range(100):
        new_perm = adjacent(route)
        temp_list[i] = route_dist - total_dist(new_perm, dist_mat, linear)

    mean_temp = np.abs(np.mean(temp_list))

    return -mean_temp / np.log(0.5)

def solve(dist_mat, max_iter = 100, alpha = 0.90, maximize = False, linear=False):
    # solve using simulated annealing
    n_cities = len(dist_mat)

    route = np.arange(n_cities, dtype=np.int64)
    np.random.shuffle(route)

    curr_temperature = start_temp(dist_mat, route, linear)

    iteration = 0
    interval = (int)(max_iter / 10)
    max_inner_iter = 10 * n_cities

    for iteration in range(max_iter):
        for k_iter in range(max_inner_iter):
            adj_route = adjacent(route)
            adj_dist = total_dist(adj_route, dist_mat, linear)
            route_dist = total_dist(route, dist_mat, linear)

            if adj_dist < route_dist and not maximize:
                route = adj_route
            elif adj_dist > route_dist and maximize:
                route = adj_route
            else:
                accept_p = np.exp(min(route_dist - adj_dist, adj_dist - route_dist) / curr_temperature)
                p = np.random.random()

                if p < accept_p:
                    route = adj_route

        if curr_temperature < 0.00001:
            curr_temperature = 0.00001
        else:
            curr_temperature *= alpha

    return route, total_dist(route, dist_mat)

def main():
    num_cities = 20

    cities = np.zeros([num_cities, 2])

    for i in range(num_cities):
        cities[i] = [np.random.rand() * 10, np.random.rand() * 10]

    dist_map = np.zeros([num_cities, num_cities])

    for i in range(num_cities):
        for j in range(num_cities):
            dist_map[i, j] = np.sqrt((cities[i, 0] - cities[j, 0])**2 + (cities[i, 1] - cities[j, 1])**2)

    print(cities)
    print(dist_map)
    route = solve(dist_map, maximize = True)
    print(route)

    #print("Correct simulated annealing")
    #print(solve_tsp_simulated_annealing(dist_map))

if __name__ == "__main__":
    main()

