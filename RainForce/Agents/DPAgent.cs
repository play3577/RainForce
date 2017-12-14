using System.Collections.Generic;
using RainForce.Models;
using RainForce.Utilities;

namespace RainForce.Agents
{
    public abstract class DPAgent
    {
        public TrainingOptions Options { get; private set; }
        public double[]Values { get;private set; }
        public double[] Policy { get; private set; }
        public int NumberOfStates { get; private set; }
        public int NumberOfActions { get; private set; }

        /// <summary>
        /// Initialize or completely reset your agent
        /// </summary>
        /// <param name="numberOfStates"></param>
        /// <param name="numberOfActions"></param>
        /// <param name="gamma"></param>
        public void Reset(int numberOfStates, int numberOfActions, double gamma = 0.75)
        {
            Options = new TrainingOptions(gamma);
            NumberOfStates = numberOfStates;
            NumberOfActions = numberOfActions;
            Policy=new double[numberOfActions*numberOfStates];
            for (int s = 0; s < NumberOfStates; s++)
            {
                var poss = GetAllowedActions(s);
                for (int i = 0, n = poss.Length; i < n; i++)
                {
                    Policy[poss[i] * NumberOfStates + s] = 1.0 / poss.Length;
                }
            }
        }
        /// <summary>
        /// You should override this method to set your allowed actions for being in state s
        /// By default actions are 0,1,2,3 regardless of s
        /// </summary>
        /// <returns></returns>
        protected virtual int[] GetAllowedActions(int s)
        {
            var ck = new int[4];
            for (int i = 0; i < ck.Length; i++)
            {
                ck[i] = i;
            }
            return ck;
        }

        /// <summary>
        /// You should override this method to get your state distro for beign in state s and taking action a
        /// default is 1 regardless of s and a
        /// </summary>
        /// <param name="s">State</param>
        /// <param name="a">Action</param>
        /// <returns></returns>
        protected virtual int NextStateDistribution(int s,int a)
        {
            return 1;
        }

        /// <summary>
        /// You should override this method to get your reward for beign in state s, taking action a in distro d
        /// default is 1 regardless of a,a or d
        /// </summary>
        /// <param name="s">State</param>
        /// <param name="a">Action</param>
        /// <param name="d">Distribution</param>
        /// <returns></returns>
        protected virtual int Reward(int s, int a, int d)
        {
            return 1;
        }

        public int Act(int state)
        {
            // behave according to the learned policy
            var poss = GetAllowedActions(state);
            var ps = new List<double>();
            for (int i = 0, n = poss.Length; i < n; i++)
            {
                var a = poss[i];
                var prob = Policy[a * NumberOfStates + state];
                ps.Add(prob);
            }
            var maxi = Util.SampleWeightedAction(ps);
            return poss[maxi];
        }

        public void Learn()
        {
            EvaluatePolicy();
            UpdatePolicy();
        }

        private void EvaluatePolicy()
        {
            // perform a synchronous update of the value function
            var Vnew = Util.ArrayOfZeros(this.NumberOfStates);
            for (var s = 0; s < this.NumberOfStates; s++)
            {
                // integrate over actions in a stochastic policy
                // note that we assume that policy probability mass over allowed actions sums to one
                var v = 0.0;
                var poss = GetAllowedActions(s);
                for (int i = 0, n = poss.Length; i < n; i++)
                {
                    var a = poss[i];
                    var prob = Policy[a * this.NumberOfStates + s]; // probability of taking action under policy
                    if (prob == 0)
                    {
                        continue;
                    } // no contribution, skip for speed
                    var ns = NextStateDistribution(s, a);
                    var rs = Reward(s, a, ns); // reward for s->a->ns transition
                    v += prob * (rs + Options.Gamma * Values[ns]);
                }
                Vnew[s] = v;
            }
            Values = Vnew; // swap
        }

        private void UpdatePolicy()
        {
            // update policy to be greedy w.r.t. learned Value function
            for (var s = 0; s < this.NumberOfStates; s++)
            {
                var poss = GetAllowedActions(s);
                // compute value of taking each allowed action
                var vmax=0.0;var nmax=0;
                var vs = new List<double>();
                for (int i = 0, n = poss.Length; i < n; i++)
                {
                    var a = poss[i];
                    var ns = NextStateDistribution(s, a);
                    var rs = Reward(s, a, ns);
                    var v = rs + Options.Gamma * Values[ns];
                    vs.Add(v);
                    if (i == 0 || v > vmax)
                    {
                        vmax = v;
                        nmax = 1;
                    }
                    else if (v == vmax)
                    {
                        nmax += 1;
                    }
                }
                // update policy smoothly across all argmaxy actions
                for (int i = 0, n = poss.Length; i < n; i++)
                {
                    var a = poss[i];
                    Policy[a * NumberOfStates + s] = (vs[i] == vmax) ? 1.0 / nmax : 0.0;
                }
            }
        }
    }
}
