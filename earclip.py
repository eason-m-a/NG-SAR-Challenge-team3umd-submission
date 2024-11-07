import numpy as np

"""
Code modified from:
https://sparkbyexamples.com/python-tutorial/implement-python-doubly-linked-list/#:~:text=Double%20Linked%20list%20is%20a,node%20in%20the%20linked%20list.
"""


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None


class DoubleLinkedList:

    def __init__(self):
        self.size = 0
        self.first = None

    def append(self, data):
        if self.first is None:
            self.first = Node(data)
            self.first.next = self.first
            self.first.prev = self.first
        else:
            temp = Node(data)
            temp.prev = self.first.prev
            temp.next = self.first

            self.first.prev.next = temp
            self.first.prev = temp

        self.size += 1

    def remove(self, data):
        if self.first is None:
            return

        curr = self.first
        for i in range(self.size):
            if curr.data == data:
                curr.prev.next = curr.next
                curr.next.prev = curr.prev

                self.size -= 1
                if self.size == 0:
                    self.first = None
                return

            curr = curr.next


def cut_polygon(coords: np.array):
    def area_of_triangle(coords: np.array) -> float:
        norms = np.linalg.norm(coords, axis=1)
        heron_sum = np.sum(norms) / 2
        process = norms * -1 + heron_sum
        mult = np.prod(process)
        area = np.sqrt(heron_sum * mult)

        return area

    def triangulate_vertices(vertices: np.array):
        indexes = triangulate(vertices)
        points = vertices[indexes].tolist()
        points = [np.array(p) for p in points]
        areas = [area_of_triangle(p) for p in points]
        a = np.sum(areas)

        return points, a

    points1, a1 = triangulate_vertices(coords)
    points2, a2 = triangulate_vertices(coords[::-1])

    result = points1 if a1 < a2 else points2

    return result


def triangulate(vertices):
    """
    Triangulation of a polygon in 2D.
    Assumption that the polygon is simple, i.e has no holes, is closed and
    has no crossings and also that its vertex order is counter clockwise.
    """
    countdown = 100000
    c = countdown

    n, m = vertices.shape
    indices = np.zeros([n - 2, 3], dtype=int)

    vertlist = DoubleLinkedList()
    for i in range(0, n):
        vertlist.append(i)
    index_counter = 0

    # Simplest possible algorithm. Create list of indexes.
    # Find first ear vertex. Create triangle. Remove vertex from list
    # Do this while number of vertices > 2.
    node = vertlist.first
    while vertlist.size > 2:
        i = node.prev.data
        j = node.data
        k = node.next.data

        vert_prev = vertices[i, :]
        vert_current = vertices[j, :]
        vert_next = vertices[k, :]

        is_convex = isConvex(vert_prev, vert_current, vert_next)
        is_ear = True
        if is_convex:
            test_node = node.next.next
            while test_node != node.prev and is_ear:
                vert_test = vertices[test_node.data, :]
                is_ear = not insideTriangle(
                    vert_prev, vert_current, vert_next, vert_test
                )
                test_node = test_node.next
        else:
            is_ear = False

        if is_ear:
            indices[index_counter, :] = np.array([i, j, k], dtype=int)
            index_counter += 1
            vertlist.remove(node.data)
            c = countdown

        node = node.next

        c -= 1
        if c == 0:
            return triangulate(vertices[::-1])

    return indices


def angleCCW(a, b):
    """
    Counter clock wise angle (radians) from normalized 2D vectors a to b
    """
    dot = a[0] * b[0] + a[1] * b[1]
    det = a[0] * b[1] - a[1] * b[0]
    angle = np.arctan2(det, dot)
    if angle < 0.0:
        angle = 2.0 * np.pi + angle
    return angle


def isConvex(vertex_prev, vertex, vertex_next):
    """
    Determine if vertex is locally convex.
    """
    a = vertex_prev - vertex
    b = vertex_next - vertex
    internal_angle = angleCCW(b, a)
    return internal_angle <= np.pi


def insideTriangle(a, b, c, p):
    """
    Determine if a vertex p is inside (or "on") a triangle made of the
    points a->b->c
    http://blackpawn.com/texts/pointinpoly/
    """

    # Compute vectors
    v0 = c - a
    v1 = b - a
    v2 = p - a

    # Compute dot products
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    # Compute barycentric coordinates
    denom = dot00 * dot11 - dot01 * dot01
    if abs(denom) < 1e-20:
        return True
    invDenom = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * invDenom
    v = (dot00 * dot12 - dot01 * dot02) * invDenom

    # Check if point is in triangle
    return (u >= 0) and (v >= 0) and (u + v < 1)
