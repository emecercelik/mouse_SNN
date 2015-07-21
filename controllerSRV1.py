# =================================================================================================================================================
#                                       Import modules
import sys
sys.path.append('/home/ercelik/opt1/nest/lib/python3.4/site-packages/')
import nest
import pickle
import random
import numpy as np

def PickleIt(data,fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def GetPickle(fileName):
    with open(fileName, 'rb') as f:
        data = pickle.load(f)
    return data

def RandomConnect(Source,Dest,numCon,Weight,Delay,Model):
    #SourceNeurons=random.sample(Source,numCon)
    for i in range(len(Source)):
        DestNeurons=random.sample(Dest,numCon)
        nest.Connect(Source[i:i+1],DestNeurons,{'rule':'all_to_all'},{'weight':Weight,'model':Model,'delay':Delay})
def PoissonRate(poi,rate):
    nest.SetStatus(poi,"rate",rate+0.0)

def SpikeNum(SpikeGen,Stime,Ftime):
    timeArray=nest.GetStatus(SpikeGen,"events")[0]['times']
    betweenTimeInt=[j for j in range(len(timeArray)) \
                    if (timeArray[j]>=Stime and timeArray[j]<=Ftime)]
    return len(betweenTimeInt)+.0

def SetNeuronInput(Neuron,Sti):
    nest.SetStatus(Neuron,"I_e",Sti)

def CalcPoiRate(coeff,offset,maxRate,minRate,value):
    return np.exp(-coeff*(value+offset)**2)*(maxRate-minRate)+minRate

def ActFunc(x):
    return -0.135*x*np.exp(0.05*x)#-0.675*x*np.exp(0.25*x)#

def InputFunc(n,maxim,minim,value,off):
    return np.exp((-n/(3*(maxim-minim)))*(value-off)**2)

def InputAcc(value):
    global nPoi,minPoiRate,maxPoiRate,maxAcc,minAcc
    n=nPoi
    step=(maxAcc-minAcc)/n
    coeff=1/(step*4.)
    offset=np.arange(minAcc,maxAcc,step)+step/2.
    Poi=[CalcPoiRate(coeff,offset[i],maxPoiRate,minPoiRate,value) for  i in range(n)]
    return Poi


bpy.context.scene.game_settings.fps=50.
dt=1000./bpy.context.scene.game_settings.fps


#nest.sli_func('synapsedict info')
# =================================================================================================================================================
#                                       Creating muscles

#~ muscle_ids = {}
#~ [ muscle_ids["forearm.L_FLEX"], muscle_ids["forearm.L_EXT"] ] = setTorqueMusclePair(reference_object_name = "obj_forearm.L",  attached_object_name = "obj_upper_arm.L",  maxF = 5.0)

#~ servo_ids = {}
#~ servo_ids["forearm.L"] = setVelocityServo(reference_object_name = "obj_forearm.L",  attached_object_name = "obj_upper_arm.L",  maxV = 10.0)
PP=60.

servo_ids = {}
servo_ids["wrist.L"]      = setPositionServo(reference_object_name = "obj_wrist.L",      attached_object_name = "obj_forearm.L", P = PP)
servo_ids["wrist.R"]      = setPositionServo(reference_object_name = "obj_wrist.R",      attached_object_name = "obj_forearm.R", P = PP)
servo_ids["forearm.L"]    = setPositionServo(reference_object_name = "obj_forearm.L",    attached_object_name = "obj_upper_arm.L", P = PP)
servo_ids["forearm.R"]    = setPositionServo(reference_object_name = "obj_forearm.R",    attached_object_name = "obj_upper_arm.R", P = PP)
servo_ids["upper_arm.L"]  = setPositionServo(reference_object_name = "obj_upper_arm.L",  attached_object_name = "obj_shoulder.L", P = PP)
servo_ids["upper_arm.R"]  = setPositionServo(reference_object_name = "obj_upper_arm.R",  attached_object_name = "obj_shoulder.R", P = PP)
servo_ids["shin_lower.L"] = setPositionServo(reference_object_name = "obj_shin_lower.L", attached_object_name = "obj_shin.L", P = PP)
servo_ids["shin_lower.R"] = setPositionServo(reference_object_name = "obj_shin_lower.R", attached_object_name = "obj_shin.R", P = PP)
servo_ids["shin.L"]       = setPositionServo(reference_object_name = "obj_shin.L",       attached_object_name = "obj_thigh.L", P = PP)
servo_ids["shin.R"]       = setPositionServo(reference_object_name = "obj_shin.R",       attached_object_name = "obj_thigh.R", P = PP)
servo_ids["thigh.L"]       = setPositionServo(reference_object_name = "obj_thigh.L",     attached_object_name = "obj_hips", P = PP)
servo_ids["thigh.R"]       = setPositionServo(reference_object_name = "obj_thigh.R",     attached_object_name = "obj_hips", P = PP)


# =================================================================================================================================================
#                                       Network creation
## Nest Kernel Initialization
T=8
nest.ResetKernel()
nest.SetKernelStatus({"overwrite_files": True,  "print_time": True})
nest.SetKernelStatus({"local_num_threads": T})
aa=np.random.randint(1,100)
nest.SetKernelStatus({'rng_seeds' : range(aa,aa+T)})
nest.sr("M_ERROR setverbosity")

## Network Parameters
neuronModel='aeif_cond_exp'
numSC=100
dx=10
dy=5
dz=2
dim=3
inpDim=6
connProb=.1
synType='tsodyks2_synapse'
numInp=(numSC*1.)
numOut=12
Weight=10.

## Input Parameters
maxPoiRate=1000.
minPoiRate=10.
nPoi=20
maxAcc=30.
minAcc=-30.

## Network Definitions and Connections
conn=[(i-1,i+1,i-dx,i+dx,i-dx*dy,i+dx*dy) for i in range(numSC)] # nor recursive conn
SpinCord=nest.Create(neuronModel,numSC)

for i in range(numSC):
    for j in range(dim*2):
        if np.random.rand()<connProb and conn[i][j]>=0 and conn[i][j]<numSC:
            nest.Connect(i+1,conn[i][j]+1,{'weight':Weight,'delay':1.,'model':synType})

    ## IDs of Input and Output Neurons
InpNeurons=tuple(np.random.randint(np.amin(SpinCord),np.amax(SpinCord)+1,size=(1,numInp))[0])
OutNeurons=tuple(np.random.randint(np.amin(SpinCord),np.amax(SpinCord)+1,size=(1,numOut))[0])
    ## Recording Devices
outSpikes=nest.Create("spike_detector", 1, {"to_file": False})
nest.ConvergentConnect(OutNeurons, outSpikes, model="static_synapse")
    ## Input Stimulus 
poisson = nest.Create( 'poisson_generator' , nPoi*inpDim , { 'rate' : minPoiRate }) # for Inp1
nest.Connect(poisson,InpNeurons[:],{'rule':'all_to_all'},{'weight':40.,'model':'static_synapse','delay':1.})

## Output Weights
wout=np.random.rand(numOut,numOut)

## Reinforcement Learning Param
wV=np.random.rand(numOut,1)
vOld=.1
count=0
eps=1.

ax=.1
ay=.1
az=.1

ax2=.1
ay2=.1
az2=.1

ax3=.1
ay3=.1
az3=.1


# =================================================================================================================================================
#                                       Evolve function
def evolve():
    #~ print("Step:", i_bl, "  Time:", t_bl)
    # ------------------------------------- Visual ------------------------------------------------------------------------------------------------
    #visual_array     = getVisual(camera_name = "Meye", max_dimensions = [256,256])
    #scipy.misc.imsave("test_"+('%05d' % (i_bl+1))+".png", visual_array)
    # ------------------------------------- Olfactory ---------------------------------------------------------------------------------------------
    olfactory_array  = getOlfactory(olfactory_object_name = "obj_nose", receptor_names = ["smell1", "plastic1"])
    # ------------------------------------- Taste -------------------------------------------------------------------------------------------------
    taste_array      = getTaste(    taste_object_name =     "obj_mouth", receptor_names = ["smell1", "plastic1"], distance_to_object = 1.0)
    # ------------------------------------- Vestibular --------------------------------------------------------------------------------------------
    vestibular_array = getVestibular(vestibular_object_name = "obj_head")
    #print (vestibular_array)
    # ------------------------------------- Sensory -----------------------------------------------------------------------------------------------
    # ------------------------------------- Proprioception ----------------------------------------------------------------------------------------
    #~ spindle_FLEX = getMuscleSpindle(control_id = muscle_ids["forearm.L_FLEX"])
    #~ spindle_EXT  = getMuscleSpindle(control_id = muscle_ids["forearm.L_EXT"])
    # ------------------------------------- Neural Simulation -------------------------------------------------------------------------------------
    # nest.Simulate()
    
    global dt,numOut,wout,wV,vOld,count,eps,ax,ay,az,ax2,ay2,az2,ax3,ay3,az3
    #nest.SetStatus(poisson,InputAcc(vestibular_array[3:6]))
    rates=np.array(InputAcc(np.hstack((vestibular_array[0:3],[ax,ay,az]))))
    rates=list(rates.reshape(1,rates.shape[0]*rates.shape[1])[0])
    nest.SetStatus(poisson,'rate',rates)
    nest.Simulate(dt)

    ## Get Events
    T_times=nest.GetStatus(outSpikes,'events')[0]['times']
    T_senders=nest.GetStatus(outSpikes,'events')[0]['senders']
    ## Time Shifting and Activity Calc
    time=nest.GetKernelStatus()['time']
    T_times-=time
    T_act=ActFunc(T_times)

    ## Output to Limbs
    outV=[[T_act[j] for j in range(T_senders.size) if OutNeurons[i]==T_senders[j]]for i in range(numOut)]
    
    outV=np.array([sum(outV[i]) for i in range(numOut)])
    outV=outV/(numOut+.0)

##    tempOutV=outV
##    vN=wV.T.dot(outV)/numOut
##    
##    outV=wout.dot(outV)
    outV=np.array([.5*np.random.rand() for i in range(numOut)])
    outV=0.5 + 0.5*np.sin(outV*t_bl)
    #outV=wout.dot(outV)/numOut
    
##    outV=np.array([np.average(outV[i]) if len(outV[i]) is not 0 else 0. for i in range(numOut)])
                    
    #print(outV)

    ## Delete previous spikes
    nest.SetStatus(outSpikes, 'n_events', 0)
    
    
    # ------------------------------------- Muscle Activation -------------------------------------------------------------------------------------


    #outV=outV/max(outV)
    #print(outV)

    controlActivity(control_id = servo_ids["wrist.L"], control_activity = outV[0])
    controlActivity(control_id = servo_ids["wrist.R"], control_activity = outV[1])
    controlActivity(control_id = servo_ids["forearm.L"], control_activity = outV[2])
    controlActivity(control_id = servo_ids["forearm.R"], control_activity = outV[3])
    controlActivity(control_id = servo_ids["upper_arm.L"], control_activity = outV[4])
    controlActivity(control_id = servo_ids["upper_arm.R"], control_activity = outV[5])
    controlActivity(control_id = servo_ids["shin_lower.L"], control_activity = outV[6])
    controlActivity(control_id = servo_ids["shin_lower.R"], control_activity = outV[7])
    controlActivity(control_id = servo_ids["shin.L"], control_activity = outV[8])
    controlActivity(control_id = servo_ids["shin.R"], control_activity = outV[9])
    controlActivity(control_id = servo_ids["thigh.L"], control_activity = outV[10])
    controlActivity(control_id = servo_ids["thigh.R"], control_activity = outV[11])

##    speed_ = 6.0
##    act_tmp         = 0.5 + 0.5*np.sin(speed_*t_bl)
##    anti_act_tmp    = 1.0 - act_tmp
##    act_tmp_p1      = 0.5 + 0.5*np.sin(speed_*t_bl - np.pi*0.5)
##    anti_act_tmp_p1 = 1.0 - act_tmp_p1
##    act_tmp_p2      = 0.5 + 0.5*np.sin(speed_*t_bl + np.pi*0.5)
##    anti_act_tmp_p2 = 1.0 - act_tmp_p2
##
##
##    controlActivity(control_id = servo_ids["wrist.L"], control_activity = 0.4)
##    controlActivity(control_id = servo_ids["wrist.R"], control_activity = 0.4)
##    controlActivity(control_id = servo_ids["forearm.L"], control_activity = 0.8*act_tmp)
##    controlActivity(control_id = servo_ids["forearm.R"], control_activity = 0.8*anti_act_tmp)
##    controlActivity(control_id = servo_ids["upper_arm.L"], control_activity = 1.0*act_tmp_p1)
##    controlActivity(control_id = servo_ids["upper_arm.R"], control_activity = 1.0*anti_act_tmp_p1)
##    controlActivity(control_id = servo_ids["shin_lower.L"], control_activity = 0.8*anti_act_tmp)
##    controlActivity(control_id = servo_ids["shin_lower.R"], control_activity = 0.8*act_tmp)
##    controlActivity(control_id = servo_ids["shin.L"], control_activity = 0.5*anti_act_tmp_p1)
##    controlActivity(control_id = servo_ids["shin.R"], control_activity = 0.5*act_tmp_p1)
##    controlActivity(control_id = servo_ids["thigh.L"], control_activity = 0.5*anti_act_tmp)
##    controlActivity(control_id = servo_ids["thigh.R"], control_activity = 0.5*act_tmp)

    

