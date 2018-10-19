import glob
from event_plot import plot_event
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import defaults

def lin_func(X, beta):
    return X.dot(beta)

def linear_fit_1D(x, y, weights):
    X = np.ones((x.shape[0], 2))
    X[:, 1] = x
    X = weights.reshape((-1,1)) * X
    beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(weights * y)
    return beta

def linear_fit(evt, threshold=defaults.DEFAULT_BASELINE, detector_dimensions=defaults.DEFAULT_DETECTOR_DIMENSIONS):
    interaction_mask = evt > threshold
    pnts = np.where(interaction_mask)
    X_inds, Y_inds, Z_inds = pnts
    X = X_inds * detector_dimensions[0] / (evt.shape[0] - 1)
    Y = Y_inds * detector_dimensions[1] / (evt.shape[1] - 1)
    Z = Z_inds * detector_dimensions[2] / (evt.shape[2] - 1)
    weights = evt[X_inds, Y_inds, Z_inds]
    beta_x = linear_fit_1D(Z, X, weights)
    beta_y = linear_fit_1D(Z, Y, weights)

    def vec_func(Z):
        pnts = np.zeros((Z.shape[0], 2))
        pnts[:,0] = beta_x[0] + beta_x[1] * Z
        pnts[:,1] = beta_y[0] + beta_y[1] * Z
        return pnts
    return beta_x, beta_y, vec_func

def convert_betas_to_angles(beta_x, beta_y):
    theta = np.arccos(1/np.sqrt(beta_x[1]**2 + beta_y[1]**2 + 1))# - np.pi/2
    phi = np.arctan(beta_y[1] / beta_x[1]) + np.pi/2 + (beta_x[1] < 0) * np.pi
    return np.asarray([theta, phi])

def get_angular_distribution(data_path, threshold = defaults.DEFAULT_BASELINE, detector_dimensions = defaults.DEFAULT_DETECTOR_DIMENSIONS, show_plots=False):
    event_files = glob.glob(data_path+'*.npy')

    angles = np.zeros((len(event_files), 2))
    for index, file in enumerate(event_files):
        if index % defaults.PRINT_EVNO_EVERY == 0:
            print("Proccessing interesting events %d - %d" % (index, index + defaults.PRINT_EVNO_EVERY))
        filename = file.split('/')[-1].split('.')[0]
        event_num = int(filename.split('_')[0])
        evt = np.load(file)

        try:
            beta_x, beta_y, vec_func = linear_fit(evt, threshold=threshold, detector_dimensions=detector_dimensions)
            angs = convert_betas_to_angles(beta_x, beta_y)
            angles[index] = angs
            #print('beta_x:', beta_x, 'beta_y:', beta_y, 'theta:', angs[0], 'phi:', angs[1])
        except Exception as e:
            print('UNABLE TO LINEAR FIT. Error:', repr(e))
            if show_plots:
                plot_event(evt, title='UNABLE TO LINEAR FIT')
            angles[index] = np.nan
            continue

        if show_plots:
            z_dim = np.linspace(0, detector_dimensions[2], 100)
            X_fit, Y_fit = vec_func(z_dim).T
            ax = plot_event(evt, detector_dimensions=detector_dimensions, show=False)
            ax.plot(X_fit, Y_fit, -z_dim, color='green')
            plt.show()

    return angles

def sphere_embed(coords):
    theta, phi = coords
    return np.asarray([np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi), np.cos(theta)])

def spherical_plot(angles):
    coordinate_angles = np.meshgrid(np.linspace(0, np.pi/2, 100), np.linspace(0, 2*np.pi, 100))

    X, Y, Z = sphere_embed(angles.T)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(*sphere_embed(coordinate_angles), cstride=10, rstride=10)
    ax.scatter(X, Y, Z, color='red', marker='o')
    plt.show()

if __name__=='__main__':
    interesting_data_folder = defaults.DEFAULT_INTERESTING_DATA_FOLDER + defaults.CURRENT_FILE + '/'
    angles = get_angular_distribution(interesting_data_folder, show_plots=False)
    print(np.nanmin(angles, axis=0))
    print(np.nanmax(angles, axis=0))
    spherical_plot(angles)

    theta, phi = angles.T

    plt.hist(theta[~np.isnan(theta)], bins=100)
    plt.show()
