import itertools
import subprocess
import re
import pandas as pd
from pandas import DataFrame
import click
import os

LR = [0.01, 0.001, 0.0001]
DF = [0.3, 0.6, 0.9]
N = [500, 1000, 1500]

def getScoreAndWR(line):
    score = float(re.search('(?<=avgScore\s)[\-0-9\.]+(?=\s)', line).group())
    actions = float(re.search('(?<=avgActions\s)[\-0-9\.]+(?=\s)', line).group())
    wr = float(re.search('(?<=avgWR\s)(.*)', line).group())
    return [actions, score, wr]

def parse_output(output):
    output = output.split('\n')
    train = getScoreAndWR(output[0])
    test = getScoreAndWR(output[1])
    return train + test

def run(layout, datapath):
    path  = os.path.join(datapath, layout)
    params = list(itertools.product(LR, DF, N))
    df = DataFrame(columns=['train actions', 'train as', 'train wr', 'avg. actions', 'avg. score', 'win rate', 'lr', 'df', 'ngames'])
    for i,[lr,disfactor,n] in enumerate(params):
        cmd = ['python', 'rlMain.py', '-r' , str(True), '-l', layout, '-ng', str(n), '-df', str(disfactor), '-lr', str(lr)]
        print '\nEXECUTING: ' + ' '.join(cmd)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output = (proc.communicate()[0]).decode('utf-8')
    
        df.loc[i] =  parse_output(output) + [lr, disfactor, n]
        os.rename('rl-data/' + layout + '_logs.pickle', path + '_logs' + str(i) + '.pickle')


    df.to_csv( path + '_data.csv', index=False)


@click.command()
@click.option('--datapath', '-d', default='rl-exps/', type=str, help='Path with csv files files')
@click.option('--layout', '-l', default='smallClassic',
              type=click.Choice(['all', 'smallClassic', 'mediumClassic', 'originalClassic']),
              help='layout: smallClassic, mediumClassic, originalClassic, or "all"')
def main(datapath, layout):

    if not os.path.exists(datapath):
        os.mkdir(datapath)

    if(layout == 'smallClassic' or layout == 'all'):
        run('smallClassic', datapath)

    if(layout == 'mediumClassic' or layout == 'all'):
        run('mediumClassic', datapath)

    if(layout == 'originalClassic' or layout == 'all'):
        run('originalClassic', datapath)

if __name__ == '__main__':
    main()
