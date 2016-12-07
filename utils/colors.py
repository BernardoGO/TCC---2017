import logging as log
import utils.basic

static_colors = []

def get_spaced_colors(n):
    log.info("generating colors:" + str(n))
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

def gen_static_colors(n):
    global static_colors
    static_colors = get_spaced_colors(n)
