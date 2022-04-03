#https://github.com/Arki99/Ransac-Line-Fit/blob/master/ransac.py
import numpy as np
from matplotlib import pyplot as plot
import math

def ransac(pts_x, pts_y, n_iter=10, dist_thresh=15):

    best_m = 0
    best_c = 0
    best_count = 0

    # set up figure and ax
    fig, ax = plot.subplots(figsize=(8,8))
    ax.scatter(pts_x, pts_y, c='blue')

    plot.ion()

    for i in range(n_iter):

        print("iteration: ", str(i))
        random_x1 = 0
        random_y1 = 0
        random_x2 = 0
        random_y2 = 0

        # select two unique points
        while random_x1 == random_x2 or random_y1 == random_y2:
            index1 = np.random.choice(pts_x.shape[0])
            index2 = np.random.choice(pts_x.shape[0])
            random_x1 = pts_x[index1]
            random_y1 = pts_y[index1]
            random_x2 = pts_x[index2]
            random_y2 = pts_y[index2]

        print("random point 1: ", random_x1, random_y1)
        print("random point 2: ", random_x2, random_y2)

        # slope and intercept for the 2 points
        if random_x2 - random_x1 is 0 and random_y2 - random_y1 is not 0:
            continue
        m = (random_y2 - random_y1) / (random_x2 - random_x1)
        c = random_y1 - m * random_x1
        count = 0
        for i, value in enumerate(pts_x):

            # calculate perpendicular distance between sample line and input data points
            dist = abs(-m * pts_x[i] + pts_y[i] - c) / math.sqrt(m ** 2 + 1)

            # count the number of inliers
            if dist < dist_thresh:
                count = count + 1

        print("Number of inliers: ", count)

        # best line has the maximum number of inliers
        if count > best_count:
            best_count = count
            best_m = m
            best_c = c

        ax.scatter([random_x1, random_x2], [random_y1, random_y2], c='red')

        # draw line between points
        line = ax.plot([0, 1000], [c, m * 1000 + c], 'red')
        plot.draw()
        plot.pause(1)
        line.pop(0).remove()
        ax.scatter([random_x1, random_x2], [random_y1, random_y2], c='blue')

    print("best_line: y = {1:.2f} x + {1:.2f}".format(m, c))

    ax.plot([0, 1000], [best_c, best_m * 1000 + best_c], 'green')
    plot.ioff()
    plot.show()

# if __name__ == "__main__":
#
#     #number of data points
#     n = 50
#
#     # generate 2d points
#     noise = np.random.uniform(-100,100,n)
#     pts_x = np.linspace(100, 1000, n) + noise
#     pts_y = np.linspace(100, 600, n) + noise
#
#     ransac(pts_x, pts_y)
