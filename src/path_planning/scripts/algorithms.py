import heapq

# --------------------- Dijkstra ---------------------
# 전체 경로 생성용
def dijkstra(grid, start, end):
    H, W = grid.shape
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    dist = {start: 0}
    prev = {}
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if u == end:
            break
        if d > dist[u]:
            continue
        for dr, dc in moves:
            v = (u[0] + dr, u[1] + dc)
            if 0 <= v[0] < H and 0 <= v[1] < W and grid[v] == 0:
                nd = d + 1
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

    if end not in prev and start != end:
        return None

    path = [end]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]

# --------------------- A* --------------------- 
# Robot 우회 경로 생성 및 dijkstra 경로 생성 실패시 사용
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid, start, end):
    H, W = grid.shape
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    open_set = []
    heapq.heappush(open_set, (heuristic(start, end), 0, start, None))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        f, g, current, parent = heapq.heappop(open_set)
        if parent is not None and current not in came_from:
            came_from[current] = parent
        if current == end:
            path = [current]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            return path[::-1]

        for dr, dc in moves:
            nbr = (current[0] + dr, current[1] + dc)
            if 0 <= nbr[0] < H and 0 <= nbr[1] < W and grid[nbr] == 0:
                tg = g + 1
                if nbr not in g_score or tg < g_score[nbr]:
                    g_score[nbr] = tg
                    heapq.heappush(open_set, (tg + heuristic(nbr, end), tg, nbr, current))
    return None