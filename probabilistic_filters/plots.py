#%%

import numpy as np
import matplotlib.pyplot as plt
from tools.task import wrap_angle


plot_all = True # for debugging


def plot_info(y, x, title):
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel(y, fontsize=16)
    plt.xlabel(x, fontsize=16)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=16)

#%% md

### Errors for EKF filter

#%%

output_dir = 'output_ekf/'

input_data = np.load(output_dir + 'input_data.npy')
output_data = np.load(output_dir + 'output_data.npy')

#%%

real_path = input_data['real_robot_path']
filtered_path = output_data['mean_trajectory']
filtered_error = np.sqrt(output_data['covariance_trajectory'])

#%%

dt = 0.1 # default setting
time = np.arange(0, input_data['num_steps'] * dt, dt)

#%%

if plot_all:
    plt.plot(filtered_path[:, 0], filtered_path[:, 1], label='filtered')
    plt.plot(real_path[:, 0], real_path[:, 1], label='real')
    plot_info(y='Y', x='X', title='EKF filter')
    plt.show()

#%%
if plot_all:
    plt.figure(figsize=(16, 16))
    plt.subplot(311)
    plt.plot(time, (filtered_path[:, 0] - real_path[:, 0]), color='b')
    plt.plot(time, 3 * filtered_error[0, 0, :], color='r')
    plt.plot(time, -3 * filtered_error[0, 0, :], color='r')
    plot_info(y=r'$\hat{x} - x$', x='time', title='Error of x')

#%%

    plt.subplot(312)
    plt.plot(time, (filtered_path[:, 1] - real_path[:, 1]), color='b')
    plt.plot(time, 3 * filtered_error[1, 1, :], color='r')
    plt.plot(time, -3 * filtered_error[1, 1, :], color='r')
    plot_info(y=r'$\hat{y} - y$', x='time', title='Error of y')

#%%

    plt.subplot(313)
    plt.plot(time, (filtered_path[:, 2] - real_path[:, 2]), color='b')
    plt.plot(time, 3 * filtered_error[2, 2, :], color='r')
    plt.plot(time, -3 * filtered_error[2, 2, :], color='r')
    plot_info(y=r'$\hat{\theta} - \theta$', x='time', title=r'Error of $\theta$')
    plt.show()


#%% md

### Errors for PF filter


#%%

output_dir = 'output_pf/'

input_data = np.load(output_dir + 'input_data.npy')
output_data = np.load(output_dir + 'output_data.npy')

#%%

real_path = input_data['real_robot_path']
filtered_path = output_data['mean_trajectory']
filtered_error = np.sqrt(output_data['covariance_trajectory'])

#%%

dt = 0.1 # default setting
time = np.arange(0, input_data['num_steps'] * dt, dt)

#%%

if plot_all:
    plt.plot(filtered_path[:, 0], filtered_path[:, 1], label='filtered')
    plt.plot(real_path[:, 0], real_path[:, 1], label='real')
    plot_info(y='Y', x='X', title='EKF filter')
    plt.show()

#%%
if plot_all:
    plt.figure(figsize=(16, 16))
    plt.subplot(311)
    plt.plot(time, (filtered_path[:, 0] - real_path[:, 0]), color='b')
    plt.plot(time, 3*filtered_error[0, 0, :], color='r')
    plt.plot(time, -3*filtered_error[0, 0, :], color='r')
    plot_info(y=r'$\hat{x} - x$', x='time', title='Error of x')

#%%

    plt.subplot(312)
    plt.plot(time, (filtered_path[:, 1] - real_path[:, 1]), color='b')
    plt.plot(time, 3*filtered_error[1, 1, :], color='r')
    plt.plot(time, -3*filtered_error[1, 1, :], color='r')
    plot_info(y=r'$\hat{y} - y$', x='time', title='Error of y')

#%%
    plt.subplot(313)
    dif_angle = []
    for i in range(filtered_path[:, 2].shape[0]):
        dif_angle.append(wrap_angle(filtered_path[i, 2] - real_path[i, 2]))

    plt.plot(time, dif_angle, color='b')
    plt.plot(time, 3*filtered_error[2, 2, :], color='r')
    plt.plot(time, -3*filtered_error[2, 2, :], color='r')
    plot_info(y=r'$\hat{\theta} - \theta$', x='time', title=r'Error of $\theta$')

    plt.show()


#%% md
## D.1
### beta to zero

#%%

data = 'for_plots/'

betas = [20, 10, 5, 2]

if plot_all:

    plt.figure(figsize=(16, 16))

    for i, b in enumerate(betas):
        input_data_ekf = np.load(data + f'ekf_b{b}_input.npy')
        output_data_ekf = np.load(data + f'ekf_b{b}_output.npy')

        input_data_pf = np.load(data + f'pf_b{b}_input.npy')
        output_data_pf = np.load(data + f'pf_b{b}_output.npy')

        real_path_ekf = input_data_ekf['real_robot_path']
        filtered_path_ekf = output_data_ekf['mean_trajectory']
        filtered_error_ekf = np.sqrt(output_data_ekf['covariance_trajectory'])

        real_path_pf = input_data_pf['real_robot_path']
        filtered_path_pf = output_data_pf['mean_trajectory']
        filtered_error_pf = np.sqrt(output_data_pf['covariance_trajectory'])

        plt.subplot(221 + i)

        plt.plot(time, abs(filtered_path_ekf[:, 0] - real_path_ekf[:, 0]), label=f'ekf')
        plt.plot(time, abs(filtered_path_pf[:, 0] - real_path_pf[:, 0]), label=f'pf')

        plot_info(y=r'$\hat{x} - x$', x='time', title=fr'Error of x with $\beta$ = {b}')

    plt.show()

    plt.figure(figsize=(16, 16))

    for i, b in enumerate(betas):
        input_data_ekf = np.load(data + f'ekf_b{b}_input.npy')
        output_data_ekf = np.load(data + f'ekf_b{b}_output.npy')

        input_data_pf = np.load(data + f'pf_b{b}_input.npy')
        output_data_pf = np.load(data + f'pf_b{b}_output.npy')

        real_path_ekf = input_data_ekf['real_robot_path']
        filtered_path_ekf = output_data_ekf['mean_trajectory']
        filtered_error_ekf = np.sqrt(output_data_ekf['covariance_trajectory'])

        real_path_pf = input_data_pf['real_robot_path']
        filtered_path_pf = output_data_pf['mean_trajectory']
        filtered_error_pf = np.sqrt(output_data_pf['covariance_trajectory'])

        plt.subplot(221 + i)

        plt.plot(time, filtered_error_ekf[0, 0, :], label='ekf')
        plt.plot(time, filtered_error_pf[0, 0, :], label='pf')

        plot_info(y=r'$\sigma$', x='time', title=fr'$\sigma$ of $x$ with $\beta$ = {b}')

    plt.show()

    plt.figure(figsize=(16, 16))

    for i, b in enumerate(betas):
        input_data_ekf = np.load(data + f'ekf_b{b}_input.npy')
        output_data_ekf = np.load(data + f'ekf_b{b}_output.npy')

        input_data_pf = np.load(data + f'pf_b{b}_input.npy')
        output_data_pf = np.load(data + f'pf_b{b}_output.npy')

        real_path_ekf = input_data_ekf['real_robot_path']
        filtered_path_ekf = output_data_ekf['mean_trajectory']
        filtered_error_ekf = np.sqrt(output_data_ekf['covariance_trajectory'])

        real_path_pf = input_data_pf['real_robot_path']
        filtered_path_pf = output_data_pf['mean_trajectory']
        filtered_error_pf = np.sqrt(output_data_pf['covariance_trajectory'])

        plt.subplot(221 + i)

        plt.plot(time, filtered_error_ekf[1, 1, :], label='ekf')
        plt.plot(time, filtered_error_pf[1, 1, :], label='pf')

        plot_info(y=r'$\sigma$', x='time', title=fr'$\sigma$ of $y$ with $\beta$ = {b}')

    plt.show()

    plt.figure(figsize=(16, 16))

    for i, b in enumerate(betas):
        input_data_ekf = np.load(data + f'ekf_b{b}_input.npy')
        output_data_ekf = np.load(data + f'ekf_b{b}_output.npy')

        input_data_pf = np.load(data + f'pf_b{b}_input.npy')
        output_data_pf = np.load(data + f'pf_b{b}_output.npy')

        real_path_ekf = input_data_ekf['real_robot_path']
        filtered_path_ekf = output_data_ekf['mean_trajectory']
        filtered_error_ekf = np.sqrt(output_data_ekf['covariance_trajectory'])

        real_path_pf = input_data_pf['real_robot_path']
        filtered_path_pf = output_data_pf['mean_trajectory']
        filtered_error_pf = np.sqrt(output_data_pf['covariance_trajectory'])

        plt.subplot(221 + i)

        plt.plot(time, filtered_error_ekf[2, 2, :], label=f'ekf')
        plt.plot(time, filtered_error_pf[2, 2, :], label=f'pf')

        plot_info(y=r'$\sigma$', x='time', title=fr'$\sigma$ of $\theta$ with $\beta$ = {b}')

    plt.show()

#%%
#!!!!AS we can see behavior of all parameters are the same, so we can estimate only x
### alpha to zero

#%%

data = 'for_plots/'

alphas = ['1', '05', '01']
title_numbers = [1, 0.5, 0.1]

if plot_all:

    plt.figure(figsize=(16, 16))

    for i, a in enumerate(alphas):
        input_data_ekf = np.load(data + f'ekf_a{a}_input.npy')
        output_data_ekf = np.load(data + f'ekf_a{a}_output.npy')

        input_data_pf = np.load(data + f'pf_a{a}_input.npy')
        output_data_pf = np.load(data + f'pf_a{a}_output.npy')

        real_path_ekf = input_data_ekf['real_robot_path']
        filtered_path_ekf = output_data_ekf['mean_trajectory']
        filtered_error_ekf = np.sqrt(output_data_ekf['covariance_trajectory'])

        real_path_pf = input_data_pf['real_robot_path']
        filtered_path_pf = output_data_pf['mean_trajectory']
        filtered_error_pf = np.sqrt(output_data_pf['covariance_trajectory'])

        plt.subplot(311 + i)

        plt.plot(time, filtered_error_ekf[1, 1, :], label=f'ekf')
        plt.plot(time, filtered_error_pf[1, 1, :], label=f'pf')

        plot_info(y=r'$\sigma$', x='time', title=fr'$\sigma$ of $y$ with $\alpha$ = {title_numbers[i]}*$\alpha_0$')



    plt.show()


    plt.figure(figsize=(16, 16))

    for i, a in enumerate(alphas):
        input_data_ekf = np.load(data + f'ekf_a{a}_input.npy')
        output_data_ekf = np.load(data + f'ekf_a{a}_output.npy')

        input_data_pf = np.load(data + f'pf_a{a}_input.npy')
        output_data_pf = np.load(data + f'pf_a{a}_output.npy')

        real_path_ekf = input_data_ekf['real_robot_path']
        filtered_path_ekf = output_data_ekf['mean_trajectory']
        filtered_error_ekf = np.sqrt(output_data_ekf['covariance_trajectory'])

        real_path_pf = input_data_pf['real_robot_path']
        filtered_path_pf = output_data_pf['mean_trajectory']
        filtered_error_pf = np.sqrt(output_data_pf['covariance_trajectory'])

        plt.subplot(311 + i)

        plt.plot(time, abs(filtered_path_ekf[:, 1] - real_path_ekf[:, 1]), label=f'ekf')
        plt.plot(time, abs(filtered_path_pf[:, 1] - real_path_pf[:, 1]), label=f'pf')


        plot_info(y=r'$\hat{y} - y$', x='time', title=fr'Error of y with a = {title_numbers[i]} * $\alpha_0$')

    plt.show()

## particle affected less

#%% md

## D.2

#%%

plt.figure(figsize=(16, 16))

data = 'for_plots/'

part = ['1', '05', '01']
title_numbers = [1, 0.5, 0.1]

if plot_all:

    for i, p in enumerate(part):

        input_data_pf = np.load(data + f'pf_p{p}_input.npy')
        output_data_pf = np.load(data + f'pf_p{p}_output.npy')

        real_path_pf = input_data_pf['real_robot_path']
        filtered_path_pf = output_data_pf['mean_trajectory']
        filtered_error_pf = np.sqrt(output_data_pf['covariance_trajectory'])

        plt.plot(time, abs(filtered_path_pf[:, 0] - real_path_pf[:, 0]), label=f'number of particles = {title_numbers[i]} * initial number of particles')

        plot_info(y=r'$\hat{x} - x$', x='time', title=fr'Error of x with a = {title_numbers[i]} * $\alpha_0$')


    plt.show()

#with dicreasing of numbers of particles errors grows

#%% md

## D.3
### ekf: underestimating of b

#%%

betas = [20, 10, 2]

if plot_all:

    plt.figure(figsize=(16, 16))
    plt.subplot(211)
    for i, b in enumerate(betas):
        input_data_ekf = np.load(data + f'ekf_bu{b}_input.npy')
        output_data_ekf = np.load(data + f'ekf_bu{b}_output.npy')

        real_path_ekf = input_data_ekf['real_robot_path']
        filtered_path_ekf = output_data_ekf['mean_trajectory']
        filtered_error_ekf = np.sqrt(output_data_ekf['covariance_trajectory'])

        plt.plot(time, abs(filtered_path_ekf[:, 0] - real_path_ekf[:, 0]), label=f'beta = {b}')

    plot_info(y=r'$\hat{x} - x$', x='time', title=fr'Error of x')


    plt.subplot(212)
    for i, b in enumerate(betas):
        input_data_ekf = np.load(data + f'ekf_bu{b}_input.npy')
        output_data_ekf = np.load(data + f'ekf_bu{b}_output.npy')

        real_path_ekf = input_data_ekf['real_robot_path']
        filtered_path_ekf = output_data_ekf['mean_trajectory']
        filtered_error_ekf = np.sqrt(output_data_ekf['covariance_trajectory'])

        plt.plot(time, filtered_error_ekf[0, 0, :], label=fr'$\sigma$ for beta = {b}')

    plot_info(y=r'$\sigma$', x='time', title=fr'$\sigma$ of x')
    plt.show()

#ekf not such big error

# %% md

### pf: underestimating of b

# %%

betas = [20, 10, 2]

if plot_all:

    plt.figure(figsize=(16, 16))
    plt.subplot(211)
    for i, b in enumerate(betas):

        input_data_pf = np.load(data + f'pf_bu{b}_input.npy')
        output_data_pf = np.load(data + f'pf_bu{b}_output.npy')

        real_path_pf = input_data_pf['real_robot_path']
        filtered_path_pf = output_data_pf['mean_trajectory']
        filtered_error_pf = np.sqrt(output_data_pf['covariance_trajectory'])

        plt.plot(time, abs(filtered_path_pf[:, 0] - real_path_pf[:, 0]), label=f'beta = {b}')

    plot_info(y=r'$\hat{x} - x$', x='time', title=fr'Error of x with $\beta$ = {b}')

    plt.subplot(212)
    for i, b in enumerate(betas):
        input_data_pf = np.load(data + f'pf_bu{b}_input.npy')
        output_data_pf = np.load(data + f'pf_bu{b}_output.npy')

        real_path_pf = input_data_pf['real_robot_path']
        filtered_path_pf = output_data_pf['mean_trajectory']
        filtered_error_pf = np.sqrt(output_data_pf['covariance_trajectory'])

        plt.plot(time, filtered_error_pf[0, 0, :], label=fr'$\sigma$ for beta = {b}')

    plot_info(y=r'$\sigma$', x='time', title=fr'$\sigma$ of x')
    plt.show()

#underestimating of sensor noise very crusual for PF

#%% md

## D.3
### ekf: overestimating of b

#%%

betas = [20, 30, 50]

if plot_all:

    plt.figure(figsize=(16, 16))

    for i, b in enumerate(betas):
        input_data_ekf = np.load(data + f'ekf_bu{b}_input.npy')
        output_data_ekf = np.load(data + f'ekf_bu{b}_output.npy')

        real_path_ekf = input_data_ekf['real_robot_path']
        filtered_path_ekf = output_data_ekf['mean_trajectory']
        filtered_error_ekf = np.sqrt(output_data_ekf['covariance_trajectory'])

        plt.plot(time, abs(filtered_path_ekf[:, 0] - real_path_ekf[:, 0]), label=f'beta = {b}')
        plt.plot(time, 3*filtered_error_ekf[0, 0, :], label=fr'3-$\sigma$ for beta = {b}')

    plot_info(y=r'$\hat{x} - x$', x='time', title=fr'Error of x')
    plt.show()

#error and covariance grows, but still work

# %% md

### pf: underestimating of b

# %%

betas = [20, 30, 50]

if plot_all:

    plt.figure(figsize=(16, 16))
    plt.subplot(211)
    for i, b in enumerate(betas):

        input_data_pf = np.load(data + f'pf_bu{b}_input.npy')
        output_data_pf = np.load(data + f'pf_bu{b}_output.npy')

        real_path_pf = input_data_pf['real_robot_path']
        filtered_path_pf = output_data_pf['mean_trajectory']
        filtered_error_pf = np.sqrt(output_data_pf['covariance_trajectory'])

        plt.plot(time, abs(filtered_path_pf[:, 0] - real_path_pf[:, 0]), label=f'beta = {b}')

    plot_info(y=r'$\hat{x} - x$', x='time', title=fr'Error of x')

    plt.subplot(212)
    for i, b in enumerate(betas):
        input_data_pf = np.load(data + f'pf_bu{b}_input.npy')
        output_data_pf = np.load(data + f'pf_bu{b}_output.npy')

        real_path_pf = input_data_pf['real_robot_path']
        filtered_path_pf = output_data_pf['mean_trajectory']
        filtered_error_pf = np.sqrt(output_data_pf['covariance_trajectory'])

        plt.plot(time, filtered_error_pf[0, 0, :], label=fr'$\sigma$ for beta = {b}')

    plot_info(y=r'$\sigma$', x='time', title=fr'$\sigma$ of x')
    plt.show()

#filter almost not affected

#%% md

### ekf: underestimating of a

#%%
data = 'for_plots/'

alphas = ['1', '05', '01']
title_numbers = [1, 0.5, 0.1]

if plot_all:

    plt.figure(figsize=(16, 16))
    plt.subplot(211)
    for i, a in enumerate(alphas):
        input_data_ekf = np.load(data + f'ekf_au{a}_input.npy')
        output_data_ekf = np.load(data + f'ekf_au{a}_output.npy')

        real_path_ekf = input_data_ekf['real_robot_path']
        filtered_path_ekf = output_data_ekf['mean_trajectory']
        filtered_error_ekf = np.sqrt(output_data_ekf['covariance_trajectory'])

        plt.plot(time, abs(filtered_path_ekf[:, 0] - real_path_ekf[:, 0]), label=fr'$\alpha$ = {title_numbers[i]} * $\alpha_0$')

    plot_info(y=r'$\hat{x} - x$', x='time', title=fr'Error of x')

    plt.subplot(212)
    for i, a in enumerate(alphas):
        input_data_ekf = np.load(data + f'ekf_au{a}_input.npy')
        output_data_ekf = np.load(data + f'ekf_au{a}_output.npy')

        real_path_ekf = input_data_ekf['real_robot_path']
        filtered_path_ekf = output_data_ekf['mean_trajectory']
        filtered_error_ekf = np.sqrt(output_data_ekf['covariance_trajectory'])

        plt.plot(time, filtered_error_ekf[0, 0, :], label=fr'$\sigma$ for $\alpha$ = {title_numbers[i]} * $\alpha_0$')

    plot_info(y=r'$\sigma$', x='time', title=fr'$\sigma$ of $x$')
    plt.show()

#error slightly grows up, while covarince grows down


#%% md

### pf: underestimating of a

#%%
data = 'for_plots/'

alphas = ['1', '05', '01']
title_numbers = [1, 0.5, 0.1]

if plot_all:

    plt.figure(figsize=(16, 16))
    plt.subplot(211)
    for i, a in enumerate(alphas):
        input_data_pf = np.load(data + f'pf_au{a}_input.npy')
        output_data_pf = np.load(data + f'pf_au{a}_output.npy')

        real_path_pf = input_data_pf['real_robot_path']
        filtered_path_pf = output_data_pf['mean_trajectory']
        filtered_error_pf = np.sqrt(output_data_pf['covariance_trajectory'])

        plt.plot(time, abs(filtered_path_pf[:, 0] - real_path_pf[:, 0]), label=fr'$\alpha$ = {title_numbers[i]} * $\alpha_0$')

    plot_info(y=r'$\hat{x} - x$', x='time', title=fr'Error of x')

    plt.subplot(212)
    for i, a in enumerate(alphas):

        input_data_pf = np.load(data + f'pf_au{a}_input.npy')
        output_data_pf = np.load(data + f'pf_au{a}_output.npy')

        real_path_pf = input_data_pf['real_robot_path']
        filtered_path_pf = output_data_pf['mean_trajectory']
        filtered_error_pf = np.sqrt(output_data_pf['covariance_trajectory'])

        plt.plot(time, filtered_error_pf[0, 0, :], label=fr'$\sigma$ for $\alpha$ = {title_numbers[i]} * $\alpha_0$')

    plot_info(y=r'$\sigma$', x='time', title=fr'$\sigma$ of $x$')
    plt.show()

#error slightly grows up, while covarince grows down

#for both: at the alpha close to zero filters follows expected trajectory instead of real

#%% md

### ekf: overestimating of a

#%%
data = 'for_plots/'

alphas = ['1', '2', '4']
title_numbers = [1, 2, 4]

if plot_all==False:

    plt.figure(figsize=(16, 16))
    plt.subplot(211)
    for i, a in enumerate(alphas):
        input_data_ekf = np.load(data + f'ekf_au{a}_input.npy')
        output_data_ekf = np.load(data + f'ekf_au{a}_output.npy')

        real_path_ekf = input_data_ekf['real_robot_path']
        filtered_path_ekf = output_data_ekf['mean_trajectory']
        filtered_error_ekf = np.sqrt(output_data_ekf['covariance_trajectory'])

        plt.plot(time, abs(filtered_path_ekf[:, 0] - real_path_ekf[:, 0]), label=fr'$\alpha$ = {title_numbers[i]} * $\alpha_0$')

    plot_info(y=r'$\hat{x} - x$', x='time', title=fr'Error of x')

    plt.subplot(212)
    for i, a in enumerate(alphas):
        input_data_ekf = np.load(data + f'ekf_au{a}_input.npy')
        output_data_ekf = np.load(data + f'ekf_au{a}_output.npy')

        real_path_ekf = input_data_ekf['real_robot_path']
        filtered_path_ekf = output_data_ekf['mean_trajectory']
        filtered_error_ekf = np.sqrt(output_data_ekf['covariance_trajectory'])

        plt.plot(time, filtered_error_ekf[0, 0, :], label=fr'$\sigma$ for $\alpha$ = {title_numbers[i]} * $\alpha_0$')

    plot_info(y=r'$\sigma$', x='time', title=fr'$\sigma$ of $x$')
    plt.show()

#error and covariance grows, but filter still stable

#%% md

### pf: overestimating of a

#%%
data = 'for_plots/'

alphas = ['1', '2', '4']
title_numbers = [1, 2, 4]

if plot_all:

    plt.figure(figsize=(16, 16))
    plt.subplot(211)
    for i, a in enumerate(alphas):
        input_data_pf = np.load(data + f'pf_au{a}_input.npy')
        output_data_pf = np.load(data + f'pf_au{a}_output.npy')


        real_path_pf = input_data_pf['real_robot_path']
        filtered_path_pf = output_data_pf['mean_trajectory']
        filtered_error_pf = np.sqrt(output_data_pf['covariance_trajectory'])

        plt.plot(time, abs(filtered_path_pf[:, 0] - real_path_pf[:, 0]), label=fr'$\alpha$ = {title_numbers[i]} * $\alpha_0$')

    plot_info(y=r'$\hat{x} - x$', x='time', title=fr'Error of x')

    plt.subplot(212)
    for i, a in enumerate(alphas):
        input_data_pf = np.load(data + f'pf_au{a}_input.npy')
        output_data_pf = np.load(data + f'pf_au{a}_output.npy')

        real_path_pf = input_data_pf['real_robot_path']
        filtered_path_pf = output_data_pf['mean_trajectory']
        filtered_error_pf = np.sqrt(output_data_pf['covariance_trajectory'])

        plt.plot(time, filtered_error_pf[0, 0, :], label=fr'$\sigma$ for $\alpha$ = {title_numbers[i]} * $\alpha_0$')

    plot_info(y=r'$\sigma$', x='time', title=fr'$\sigma$ of $x$')

    plt.show()

#error and covariance grows, but filter still stable