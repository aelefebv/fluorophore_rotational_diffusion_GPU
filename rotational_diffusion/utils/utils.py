from numpy.random import uniform


def split_counts_xy(x, y, t):
    p_x, p_y = x**2, y**2
    r = uniform(0, 1, size=len(t))
    in_channel_x = (r < p_x)
    in_channel_y = (p_x <= r) & (r < p_x + p_y)
    t_x, t_y = t[in_channel_x], t[in_channel_y]
    return t_x, t_y
