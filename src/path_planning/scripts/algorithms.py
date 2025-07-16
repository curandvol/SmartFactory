import heapq

# --------------------- Dijkstra ---------------------
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

# --------------------- D* Lite ---------------------
# Incremental path planning: update only changed parts of the grid

INFINITY = float('inf')

class DStarLite:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.height, self.width = grid.shape
        self.start = start
        self.goal = goal
        self.k_m = 0
        self.rhs = {}
        self.g = {}
        self.U = []  # priority queue of (key, node)
        self._init_dstar()

    def _init_dstar(self):
        # Initialize g and rhs
        for r in range(self.height):
            for c in range(self.width):
                self.g[(r, c)] = INFINITY
                self.rhs[(r, c)] = INFINITY
        # Goal's rhs=0
        self.rhs[self.goal] = 0
        # Push goal into queue
        heapq.heappush(self.U, (self._calculate_key(self.goal), self.goal))
        self.last = self.start

    def _calculate_key(self, node):
        g_rhs = min(self.g[node], self.rhs[node])
        h = heuristic(self.start, node)
        return (g_rhs + h + self.k_m, g_rhs)

    def _update_vertex(self, u):
        if u != self.goal:
            # rhs = min successor cost
            self.rhs[u] = min(
                [self.g[s] + 1 for s in self._get_successors(u)]
            )
        # remove u from U if it exists
        self.U = [(k, n) for (k, n) in self.U if n != u]
        heapq.heapify(self.U)
        if self.g[u] != self.rhs[u]:
            heapq.heappush(self.U, (self._calculate_key(u), u))

    def _compute_shortest_path(self):
        while self.U:
            k_old, u = heapq.heappop(self.U)
            k_new = self._calculate_key(u)
            if k_old < k_new:
                heapq.heappush(self.U, (k_new, u))
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                for s in self._get_predecessors(u):
                    self._update_vertex(s)
            else:
                self.g[u] = INFINITY
                for s in self._get_predecessors(u) + [u]:
                    self._update_vertex(s)
            if self._calculate_key(self.start) >= self.U[0][0] and self.rhs[self.start] == self.g[self.start]:
                break

    def replan(self, start, goal):
        self.start = start
        self.goal = goal
        self._update_vertex(self.goal)
        self._compute_shortest_path()
        if self.g[self.start] == INFINITY:
            return None
        # reconstruct path
        path = [self.start]
        s = self.start
        while s != self.goal:
            succ = self._get_successors(s)
            s = min(succ, key=lambda x: self.g[x] + 1)
            path.append(s)
        return path

    def update_obstacles(self, changed_cells):
        # Inform algorithm of changed obstacle cells
        for cell in changed_cells:
            r, c = cell
            # mark as blocked
            self.grid[cell] = 1
            for u in self._get_predecessors(cell):
                self._update_vertex(u)
        self.k_m += heuristic(self.last, self.start)
        self.last = self.start

    def _get_successors(self, u):
        r, c = u
        moves = [(1,0),(-1,0),(0,1),(0,-1)]
        return [
            (nr, nc) for dr, dc in moves
            if 0 <= (nr:=r+dr) < self.height and 0 <= (nc:=c+dc) < self.width
            and self.grid[nr, nc] == 0
        ]

    def _get_predecessors(self, u):
        # On grid, predecessors == successors
        return self._get_successors(u)

__all__ = ['dijkstra', 'heuristic', 'astar', 'DStarLite']

