import os
import sys
from multiprocessing import Pool
from collections import OrderedDict


dataset = sys.argv[1]
model = sys.argv[2]
nRuns = int(sys.argv[3]) # Number of runs.
coreList = sys.argv[4].split('/') # Feed core list to run, separated by '/'. eg. 0-1/2/3,4,5/6/7
experimentDir = sys.argv[5] # Name of experiment directory where outputs of this run will be stored.

dataset2dir = {
                  'Chennai':'../datasets/traffic_tpp/chn/'
              }

model2dir = {
                'RMTPP': 'tf_rmtpp/src/'
            }

def getConfig():
    config = OrderedDict()
    if model in ['RMTPP']:
        config['--epochs'] = 50
        config['--test-eval'] = ''
        config['--init-learning-rate'] = 0.0001
        config['--cpu-only'] = ''
        #config['--normalize'] = ''

    return config

modelConfig = getConfig()

#seedList = [1,12,123,1234,12345]
#seedList = [100,200,300,400,500]
seedList = [42]
runIDList = range(nRuns)

def runCommand(command):
    os.system(command)

outputDirName = os.path.join('Outputs', experimentDir, dataset+'_'+model)
allDirs = [os.path.join('Outputs', experimentDir, i)
           for i in os.listdir(os.path.join('Outputs', experimentDir))
           if os.path.isdir(os.path.join('Outputs', experimentDir, i))]
if allDirs:
    matches = [i for i in allDirs if i.startswith(outputDirName)]
    if matches:
        ids = [int(i.split('_')[-1]) for i in matches]
        newId = max(ids) + 1
    else:
        newId = 1
else:
    newId = 1
outputDirName = outputDirName + '_' + str(newId)
if not os.path.exists(outputDirName):
    os.mkdir(outputDirName)
commandLineInput = '' # '-dataset '+dataset+' -modelToRun '+model+' ' #TODO: change this
for param, val in modelConfig.items():
    commandLineInput += ' '+param+' '+str(val)
pool = Pool(nRuns)
commandList = list()
command = 'python3.6 ' + os.path.join(model2dir[model], 'run.py') + ' ' \
                       + os.path.join(dataset2dir[dataset], 'event-train.txt') + ' ' \
                       + os.path.join(dataset2dir[dataset], 'time-train.txt') + ' ' \
                       + os.path.join(dataset2dir[dataset], 'event-test.txt') + ' ' \
                       + os.path.join(dataset2dir[dataset], 'time-test.txt') + ' ' \
                       + os.path.join(dataset2dir[dataset], 'feats-train.pkl') + ' ' \
                       + os.path.join(dataset2dir[dataset], 'feats-test.pkl') + ' ' \
                       + commandLineInput #TODO: change this
for r in range(1, nRuns+1):
    seed = str(seedList[r-1])
    outputSeedDir = os.path.join(outputDirName, 'seed_'+seed)
    if not os.path.exists(outputSeedDir):
        os.mkdir(outputSeedDir)
    outputFileName = os.path.join(outputSeedDir, 'seed_'+seed)
    gtVsPredOutputFile = os.path.join(outputSeedDir, 'seed_'+seed+'gtVsPred')
    commandList.append('taskset -c '+str(coreList[r-1])+' '+command+' --seed '+seed
                                    +' --gtVsPredOutputFile '+gtVsPredOutputFile
                                    +' --summary '+outputSeedDir
                                    +' --save '+outputSeedDir
                                    +' >>'+outputFileName)
    with open(outputFileName,'a') as outFile:
        outFile.write(command+'\n')
print(commandList)
pool.map(runCommand, commandList)
