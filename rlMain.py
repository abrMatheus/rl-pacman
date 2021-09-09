import subprocess
import re
import click
import os
import numpy as np
import pickle


def compute_score(layout, n_games, path,  args,  quiet=True, agent='AQLAgent'):
    cmd = ['python', 'pacman.py', '-n', str(n_games), '--layout', layout,  '--pacman', agent, '-a', 'path=' + path + ',' + args + '']
    if quiet:
        cmd.append('-q')
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    output = (proc.communicate()[0]).decode('utf-8')

    num_actions = [
        int(n) for n in re.findall(r'(?<=\#\sActions:\s)[0-9]+(?=\s#)', output)
    ]

    total_reward = [
        float(n) for n in re.findall(r'(?<=Reward:\s)[\-0-9]+(?=\s)', output)
    ]

    output = output.split('\n')
    avg_score = float(re.search('[\-0-9\.]+', output[-5]).group())
    win_rate = float(re.findall('[0-9\.]+', output[-3])[-1])

    scores = [float(s) for s in re.findall(r'[\-\.0-9]+', output[-4])]
    return avg_score, win_rate, scores, num_actions, total_reward


def saveValues(datapath, layout, logscore, num_actions, total_reward):
    tdict = { 'scores': logscore, 'actions': num_actions, 'reward': total_reward}

    with open(datapath + "/" + layout + '_logs.pickle', 'wb') as handle:
            pickle.dump(tdict, handle)

def parse_args(discount_factor, learning_rate):
    return 'learning_rate=' + str(learning_rate) + ',discount_factor=' + str(discount_factor)

@click.command()
@click.option('--layout', '-l', default='smallClassic', type=str, help='Pacman game layout, smallClassic as default')
@click.option('--n-games', '-ng', default=2, type=int, help='Number of games of training')
@click.option('--n-games-test', '-ngte', default=10, type=int, help='Number of games of testing')
@click.option('--quiet', '-q', default=True, type=bool, help='Quiet graphics (used for both, training and testing)')
@click.option('--reset', '-r', default=False, type=bool, help='Start a new training if True')
@click.option('--datapath', '-d', default='rl-data/', type=str, help='Path with weights .pickle files, rl-data/ as default')
@click.option('--discount-factor', '-df', default=0.9, type=float, help='Discount factor, 0.9 as default')
@click.option('--learning-rate', '-lr', default=0.001, type=float, help='Learning rate, 0.001 as default')
def main(layout, n_games, n_games_test, quiet, reset, datapath, discount_factor, learning_rate):
    
    if not os.path.exists(datapath):
        os.mkdir(datapath)

    srcDataFile = datapath + layout + ".pickle"
    dstDataFile = datapath + "run.pickle"

    if os.path.isfile(srcDataFile) and not reset:
        os.system("cp " + srcDataFile + " " + dstDataFile)
    elif os.path.isfile(dstDataFile):
        os.remove(dstDataFile)

    if n_games > 0:
        pacman_args = parse_args(discount_factor, learning_rate)
        avg_score, win_rate, logscore, num_actions, total_reward = compute_score(layout, n_games, path=datapath,
                                                                                args=pacman_args, quiet=quiet)
    
    if n_games_test > 0:
        test_args = parse_args(discount_factor, 0.0)
        test_score, test_wr, test_ls, test_na, test_reward = compute_score(layout, n_games=n_games_test, path=datapath, 
                                                    args=test_args, quiet=quiet)


    if n_games > 0:

        saveValues(datapath, layout, logscore, num_actions, total_reward)

        os.rename(dstDataFile, srcDataFile)

        print 'Training done with avgScore', avg_score, 'avgActions', np.mean(num_actions), 'avgWR', win_rate

    if n_games_test > 0:

        print 'Testing performance with avgScore', test_score, 'avgActions', np.mean(test_na), 'avgWR', test_wr



#DEMO python pacman.py -p AQLAgent -l smallClassic -a path=rl-data/,discount_factor=0.9,learning_rate=0.001
if __name__ == '__main__':
    main()
