import numpy as np
import matplotlib.pyplot as plt
import pickle
import shutil, os

with open('flight.dat', 'rb') as f:
    t, X, Y, Vx, Vy, x_start, x_end, y_start, y_end, nv, ns, limit, separation, airspace_type, aggregate = pickle.load(f)


theta = np.degrees(np.arctan2(Vy, Vx)) + 35


nn, nv = X.shape

try:
    shutil.rmtree('frames')
except FileNotFoundError:
    pass
os.makedirs('frames')


i_frame = 0
frames = 10*[0] + list(range(nn)) + 20*[nn - 1]
for i in frames:
    fig = plt.figure()
    ax = plt.gca()

    if airspace_type == 0:
        plt.plot([-1000, 1000], [1000, 1000], 'k')
        plt.plot([-1000, -1000], [1000, -1000], 'k')
        plt.plot([-1000, 1000], [-1000, -1000], 'k')
        plt.plot([1000, 1000], [-1000, 1000], 'k')
    else:
        circle = plt.Circle((0, 0), 1000, fill=False)
        plt.gca().add_artist(circle)

    plt.scatter(x_start, y_start, c='k', marker='x')
    plt.scatter(x_end, y_end, c='k')


    for k in range(nv):
        #plt.scatter(X[i, k], Y[i, k], marker=(3, 0, theta[i, k]))
        plt.scatter(X[i, k], Y[i, k])

        circle = plt.Circle((X[i, k], Y[i, k]), limit/2, fill=False)
        plt.gca().add_artist(circle)

        plt.plot(X[:i,k], Y[:i, k], 'k--', linewidth=0.25)

        #plt.plot([x_start[k], X[i, k]], [y_start[k], Y[i, k]], 'k--', linewidth=0.25)
        #plt.plot([X[i, k], x_end[k]], [Y[i, k], y_end[k]], 'k--', linewidth=0.25)
    
    plt.tight_layout(pad=1)
    plt.axis('equal')
    plt.xlim(-1200,1200)
    plt.ylim(-1200,1200)
    fig.savefig('frames/%03d.png' % i_frame, dpi=fig.dpi)

    plt.close()
    i_frame += 1

duration = 7
fps = int(i_frame / duration) + 1

cmd = "/usr/local/bin/ffmpeg -y -r %d -i frames/%%03d.png -c:v libx264 -vf fps=%d -pix_fmt yuv420p out.mp4; open out.mp4" % (fps, fps)
os.system(cmd)

