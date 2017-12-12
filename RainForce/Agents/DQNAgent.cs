using System;
using System.Collections.Generic;
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

        private Matrix previousState, _nextState;
        private double _reward, t;
        private int nextAction, previousAction;

        Graph LastGraph;

        public DQNAgent(TrainingOptions options, int states, int actions)
        {
            Net= new Network();
            NumberOfStates = states;
            NumberOfActions = actions;
            Options = options;
            Net.W1 = new Matrix(Options.HiddenUnits,NumberOfStates);
            Net.W2 = new Matrix(NumberOfActions, options.HiddenUnits);
            Net.B1 = new Matrix(options.HiddenUnits,1);
            Net.B2 = new Matrix(NumberOfActions, 1);
            Util.FillMatrixWithRandomGaussianNumbers(Net.W1,0,0.01);
            Util.FillMatrixWithRandomGaussianNumbers(Net.W2, 0, 0.01);
            Util.FillMatrixWithRandomGaussianNumbers(Net.B1, 0, 0.01);
            Util.FillMatrixWithRandomGaussianNumbers(Net.B2, 0, 0.01);
            Experience= new List<Experience>();
        }

        public Matrix Forward(Network net, Matrix state, bool backProp)
        {
            
            LastGraph = new Graph(backProp);
            var a1mat = LastGraph.AddMatrix(LastGraph.MultiplyMatrix(net.W1, state), net.B1);
            var h1mat = LastGraph.GetMatrixFromTangent(a1mat);
            var action = LastGraph.AddMatrix(LastGraph.MultiplyMatrix(net.W2, h1mat), net.B2);
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
                a = Util.RandomI(0, NumberOfActions);
            }
            else
            {
                // greedy wrt Q function
                var amat = Forward(Net, state, false);
                 a =Util.Maxi(amat.Weights); // returns index of argmax action
            }
            // shift state memory
            previousState = _nextState;
            previousAction = nextAction;
            _nextState = state;
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
                a = Util.RandomI(0, NumberOfActions);
            }
            else
            {
                // greedy wrt Q function
                var amat = Forward(Net, state, false);
                a = Util.Maxi(amat.Weights); // returns index of argmax action
            }
            // shift state memory
            previousState = _nextState;
            previousAction = nextAction;
            _nextState = state;
            nextAction = a;
            return a;
        }

        public void Learn(double reward)
        {
            // perform an update on Q function
            if (_reward != 0 && Options.Alpha> 0)
            {

                // learn from this tuple to get a sense of how "surprising" it is to the agent
                var tderror = LearnFromTuple(previousState, previousAction, _reward, _nextState);
                CurrentError = tderror; // a measure of surprise

                // decide if we should keep this experience in the replay
                if (t % Options.ExperienceAddEvery == 0)
                {
                    var exp = new Experience
                    {
                        PreviousAction = previousAction,
                        NextAction = nextAction,
                        Reward = _reward,
                        PreviousState = previousState,
                        NextState = _nextState
                    };
                   Experience.Add(exp);
                    if (Experience.Count > Options.ExperienceSize)
                    {
                        Experience.RemoveAt(0);
                    }
                }
                t += 1;

                // sample some additional experience from replay memory and learn from it
                for (var k = 0; k <Options.LearningSteps; k++)
                {
                    var ri = Util.RandomI(0, Experience.Count);
                    var e = Experience[ri];
                    LearnFromTuple(e.PreviousState,e.PreviousAction, e.Reward, e.NextState);
                }
            }
            _reward = reward; // store for next update
        }

        private double LearnFromTuple(Matrix prevState, int prevAction, double reward, Matrix nextState)
        {
            // want: Q(s,a) = r + gamma * max_a' Q(s',a')

            // compute the target Q value
            var tmat = Forward(Net, nextState, false);
            var qmax = reward + Options.Gamma * tmat.Weights[Util.Maxi(tmat.Weights)];

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
            //LastGraph.backward(); // compute gradients on net params //not needed since we already back propagated

            // update net
            Util.UpdateNetwork(Net, Options.Alpha);
            return tderror;
        }
    }
}
