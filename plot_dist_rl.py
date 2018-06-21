import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from deep_rl import *

def plot(**kwargs):
    import matplotlib.pyplot as plt
    kwargs.setdefault('average', False)
    # kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    kwargs.setdefault('top_k_perf', lambda x: np.mean(x[-20:]))
    kwargs.setdefault('max_timesteps', 1e8)
    kwargs.setdefault('episode_window', 100)
    kwargs.setdefault('x_interval', 1000)
    kwargs.setdefault('down_sample', True)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    data = plotter.load_results(names, episode_window=kwargs['episode_window'], max_timesteps=kwargs['max_timesteps'])
    print('')

    figure = kwargs['figure']
    plt.figure(figure)
    if kwargs['average']:
        color = kwargs['color']
        x, y = plotter.average(data, kwargs['x_interval'], kwargs['max_timesteps'], top_k=kwargs['top_k'],
                               top_k_perf=kwargs['top_k_perf'])
        print(y.shape)
        if kwargs['down_sample']:
            indices = np.linspace(0, len(x) - 1, 500).astype(np.int)
            x = x[indices]
            y = y[:, indices]
        name = names[0].split('/')[-1]
        sns.tsplot(y, x, condition=name, color=Plotter.COLORS[color], ci='sd')
        plt.title(names[0])
    else:
        for i, name in enumerate(names):
            x, y = data[i]
            if 'color' not in kwargs.keys():
                color = Plotter.COLORS[i]
            else:
                color = Plotter.COLORS[kwargs['color']]
            plt.plot(x, y, color=color, label=name if i==0 else '')
    plt.legend()
    if 'y_lim' in kwargs.keys():
        plt.ylim(kwargs['y_lim'])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')

def deterministic_plot(**kwargs):
    import matplotlib.pyplot as plt
    kwargs.setdefault('average', False)
    # kwargs.setdefault('color', 0)
    kwargs.setdefault('top_k', 0)
    kwargs.setdefault('top_k_perf', lambda x: np.mean(x[-20:]))
    kwargs.setdefault('max_timesteps', 1e8)
    kwargs.setdefault('episode_window', 100)
    kwargs.setdefault('x_interval', 1600)
    kwargs.setdefault('rep', 20)
    plotter = Plotter()
    names = plotter.load_log_dirs(**kwargs)
    raw_data = plotter.load_results(names, episode_window=0, max_timesteps=kwargs['max_timesteps'])
    data = []
    for x, y in raw_data:
        y = y[: len(y) // kwargs['rep'] * kwargs['rep']]
        y = np.reshape(np.asarray(y), (-1, kwargs['rep'])).mean(-1)
        x = np.arange(y.shape[0]) * kwargs['x_interval']
        data.append([x, y])
    print('')

    figure = kwargs['figure']
    plt.figure(figure)
    if kwargs['average']:
        color = kwargs['color']
        data = [plotter.window_func(x, y, kwargs['episode_window'], np.mean) for x, y in data]
        x = data[0][0]
        y = [y for x, y in data]
        y = np.stack(y)
        sns.tsplot(y, x, condition=names[0], color=Plotter.COLORS[color], ci='sd')
    else:
        for i, name in enumerate(names):
            x, y = data[i]
            if 'color' not in kwargs.keys():
                color = Plotter.COLORS[i]
            else:
                color = Plotter.COLORS[kwargs['color']]
            plt.plot(x, y, color=color, label=name if i==0 else '')
    plt.legend()
    if 'y_lim' in kwargs:
        plt.ylim(kwargs['y_lim'])
    plt.xlabel('timesteps')
    plt.ylabel('episode return')
    # plt.show()


if __name__ == '__main__':
    # game = 'Freeway'
    # game = 'Seaquest'
    # game = 'MsPacman'
    # game = 'Frostbite'
    # game = 'Enduro'
    # game = 'JourneyEscape'
    # game = 'Tennis'
    # game = 'Pong'
    # game = 'Boxing'
    # game = 'IceHockey'
    # game = 'Skiing'
    # game = 'SpaceInvaders'
    # game = 'UpNDown'
    # game = 'BeamRider'
    # game = 'Robotank'
    # game = 'BankHeist'
    # game = 'BattleZone'
    games = [
        'FreewayNoFrameskip-v4',
        'BeamRiderNoFrameskip-v4',
        'BattleZoneNoFrameskip-v4',
        'RobotankNoFrameskip-v4',
        'QbertNoFrameskip-v4',
        'AlienNoFrameskip-v4',
        'AmidarNoFrameskip-v4',
        'SeaquestNoFrameskip-v4',
        'MsPacmanNoFrameskip-v4',
        'EnduroNoFrameskip-v4',
        ]
    # games = [
    #     'BreakoutNoFrameskip-v4',
    #     'AssaultNoFrameskip-v4',
    #     'JamesbondNoFrameskip-v4',
    #     'DemonAttackNoFrameskip-v4'
    # ]

    train_kwargs = {
        'episode_window': 100,
        'top_k': 0,
        'max_timesteps': int(4e7),
        # 'max_timesteps': int(3e7),
        'average': True,
        'x_interval': 1000
    }
    test_kwargs = {
        'average': False,
        'x_interval': 16e4,
        'rep': 10,
        'max_timesteps': int(4e7),
    }

    patterns = [
        'original',
        't0b0_ns',
        't01b01_ns',
        't001b001_ns',
        # 't0b0_s',
    ]

    patterns = [
        't001b001_s_le',
        # 't001b001_s_se',
        'n_step_qr_dqn',
        'n_step_qr_le_dqn',
        # 'n_step_dqn',
        # 'n_step_dqn',
    ]

    for j, game in enumerate(games):
        for i, p in enumerate(patterns):
            plot(pattern='.*dist_rl.*%s.*%s.*train.*' % (game, p), figure=j, color=i, **train_kwargs)
            plt.savefig('data/dist_rl_images/n-step-qr-dqn-%s-train.png' % (game))
            # plot(pattern='.*dist_rl.*%s.*%s.*test.*' % (game, p), figure=j, color=i, **train_kwargs)
            # plt.savefig('data/dist_rl_images/n-step-qr-dqn-%s-test.png' % (game))
            # deterministic_plot(pattern='.*dist-rl.*%s.*%s.*test.*' % (game, p), figure=j, color=i, **test_kwargs)
            # plt.savefig('data/dist_rl_images/n-step-qr-dqn-%s-test.png' % (game))
    # plt.show()

    train_kwargs = {
        'episode_window': 5000,
        'top_k': 0,
        # 'max_timesteps': int(4e7),
        'max_timesteps': int(2e7),
        'average': True,
        'x_interval': 1000,
        # 'y_lim': [-2, 5],
        'down_sample': True,
    }
    test_kwargs = {
        'episode_window': 50,
        'average': True,
        'x_interval': 16e4,
        'rep': 10,
        'max_timesteps': int(4e7),
        'y_lim': [-2, 5]
    }

    tag='b0e03'
    patterns = [
        'original',
        't0b0e03',
        't01b0e03',
        't05b0e03',
        't09b0e03',
    ]

    tag='b0e00'
    patterns = [
        'original',
        't0b0e03',
        't01b0e03',
        't05b0e03',
        't09b0e03',
    ]

    tag = 'on-termination'
    patterns = [
        't0b0_ns',
        't001b001_ns',
        't01b01_ns',
        't0b0_s',
        't001b001_s',
        't01b01_s',
        'original',
    ]

    # for i, p in enumerate(patterns):
    #     plot(pattern='.*dist_rl-IceCliff.*%s.*-train.*' %(p), figure=0, color=i, **train_kwargs)
    #     plt.savefig('data/dist_rl_images/n-step-qr-dqn-IceCliff-%s-train.png' % (tag))
    #     plot(pattern='.*dist_rl-IceCliff.*%s.*-test.*' %(p), figure=1, color=i, **train_kwargs)
    #     plt.savefig('data/dist_rl_images/n-step-qr-dqn-IceCliff-%s-test.png' % (tag))
    #     # deterministic_plot(pattern='.*dist_rl-IceCliff.*%s.*-test.*' %(p), figure=1, color=i, **test_kwargs)
    #     # plt.savefig('data/dist_rl_images/n-step-qr-dqn-IceCliff-%s-test.png' % (train_kwargs['tag']))
    # plt.show()

    train_kwargs = {
        'average': True,
        'x_interval': 1000,
        'top_k': 0,
        'max_timesteps': int(3e5),
        'down_sample': False
    }
    test_kwargs = {
        'average': True,
        'x_interval': 1600,
        'rep': 20,
        'max_timesteps': int(3e6),
    }
    patterns = [
        'original',
        't0b0ns',
        't0b0s',
    ]
    # for i, p in enumerate(patterns):
    #     plot(pattern='.*dist_rl-CliffWalking.*bootstrapped_qr_dqn_cliff.*%s.*train.*' % (p), figure=0, color=i, **train_kwargs)
    # plt.show()

