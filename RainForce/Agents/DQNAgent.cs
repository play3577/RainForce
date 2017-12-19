using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Newtonsoft.Json;
using RainForce.Models;
using RainForce.Utilities;

namespace RainForce.Agents
{
    public class DQNAgent
    {
        public TrainingOptions Options { get; }
        public Network Net { get; set; }
        public List<Experience> Experience { get; }
        public int NumberOfStates { get; }
        public int NumberOfActions { get; }
        public double CurrentError { get; set; }

        private Matrix previousStateCache, nextStateCache;
        private double rewardCache, learnLoopCount;
        private int nextAction, previousAction;

        Graph LastGraph;

        public DQNAgent(TrainingOptions options, int states, int actions)
        {
            Net= new Network(4,"DQNAgent");
            NumberOfStates = states;
            NumberOfActions = actions;
            Options = options;
            Net.Matrices[0] = new Matrix(Options.HiddenUnits,NumberOfStates);
            Net.Matrices[1] = new Matrix(NumberOfActions, options.HiddenUnits);
            Net.Matrices[2] = new Matrix(options.HiddenUnits,1);
            Net.Matrices[3] = new Matrix(NumberOfActions, 1);
            Util.FillMatrixWithRandomGaussianNumbers(Net.Matrices[0],0,0.01);
            Util.FillMatrixWithRandomGaussianNumbers(Net.Matrices[1], 0, 0.01);
            Util.FillMatrixWithRandomGaussianNumbers(Net.Matrices[2], 0, 0.01);
            Util.FillMatrixWithRandomGaussianNumbers(Net.Matrices[3], 0, 0.01);
            Experience= new List<Experience>();
        }

        public Matrix Forward(Network net, Matrix state, bool backProp)
        {
            
            LastGraph = new Graph(backProp);
            var a1mat = LastGraph.AddMatrix(LastGraph.MultiplyMatrix(Net.Matrices[0], state), Net.Matrices[2]);
            var h1mat = LastGraph.GetMatrixFromTangent(a1mat);
            var action = LastGraph.AddMatrix(LastGraph.MultiplyMatrix(Net.Matrices[1], h1mat), Net.Matrices[3]);
            return action;
        }

        public int Act(double[] stateArray)
        {
            var r= new Random();
            // convert to a Mat column vector
            var state = new Matrix(NumberOfStates, 1);
            state.Set(stateArray);
            var a = 0;
            // epsilon greedy policy
            if (r.NextDouble() <Options.Epsilon)
            {
                a = Util.Random(0, NumberOfActions);
            }
            else
            {
                // greedy wrt Q function
                var amat = Forward(Net, state, false);
                 a =Util.ActionFromWeights(amat.Weights); // returns index of argmax action
            }
            // shift state memory
            previousStateCache = nextStateCache;
            previousAction = nextAction;
            nextStateCache = state;
            nextAction = a;
            return a;
        }

        public int Act(int[] stateArray)
        {
            var r = new Random();
            // convert to a Mat column vector
            var state = new Matrix(NumberOfStates, 1);
            state.Set(stateArray);
            var a = 0;
            var y = r.NextDouble();
            // epsilon greedy policy
            if ( y< Options.Epsilon)
            {
                a = Util.Random(0, NumberOfActions);
            }
            else
            {
                // greedy wrt Q function
                var amat = Forward(Net, state, false);
                a = Util.ActionFromWeights(amat.Weights); // returns index of argmax action
            }
            // shift state memory
            previousStateCache = nextStateCache;
            previousAction = nextAction;
            nextStateCache = state;
            nextAction = a;
            return a;
        }

        public void Learn(double reward)
        {
            // perform an update on Q function
            if (rewardCache != 0 && Options.Alpha> 0)
            {

                // learn from this tuple to get a sense of how "surprising" it is to the agent
                var tderror = LearnFromTuple(previousStateCache, previousAction, rewardCache, nextStateCache);
                CurrentError = tderror; // a measure of surprise

                // decide if we should keep this experience in the replay
                if (learnLoopCount % Options.ExperienceAddEvery == 0)
                {
                    var exp = new Experience
                    {
                        PreviousAction = previousAction,
                        NextAction = nextAction,
                        Reward = rewardCache,
                        PreviousState = previousStateCache,
                        NextState = nextStateCache
                    };
                   Experience.Add(exp);
                    if (Experience.Count > Options.ExperienceSize)
                    {
                        Experience.RemoveAt(0);
                    }
                }
                learnLoopCount += 1;

                // sample some additional experience from replay memory and learn from it
                for (var k = 0; k <Options.LearningSteps; k++)
                {
                    var ri = Util.Random(0, Experience.Count);
                    var e = Experience[ri];
                    LearnFromTuple(e.PreviousState,e.PreviousAction, e.Reward, e.NextState);
                }
            }
            rewardCache = reward; // store for next update
        }

        private double LearnFromTuple(Matrix prevState, int prevAction, double reward, Matrix nextState)
        {
            // want: Q(s,a) = r + gamma * max_a' Q(s',a')

            // compute the target Q value
            var tmat = Forward(Net, nextState, false);
            var qmax = reward + Options.Gamma * tmat.Weights[Util.ActionFromWeights(tmat.Weights)];

            // now predict
            var pred = Forward(Net, prevState, true);

            var tderror = pred.Weights[prevAction] - qmax;
            var clamp = Options.ErrorClamp;
            if (Math.Abs(tderror) > clamp)
            {  // huber loss to robustify
                if (tderror > clamp) tderror = clamp;
                if (tderror < -clamp) tderror = -clamp;
            }
            pred.BackPropWeights[prevAction] = tderror;
            LastGraph.Backward();

            // update net
            Util.UpdateNetwork(Net, Options.Alpha);
            return tderror;
        }

        public string AgentToJson()
        {
            return JsonConvert.SerializeObject(this);
        }

        public static DQNAgent AgentFromJson(string agent)
        {
           return JsonConvert.DeserializeObject<DQNAgent>(agent);
        }
    }
}
