using System;
using System.Collections.Generic;
using System.Linq;
using RainForce.Models;

namespace RainForce.Utilities
{
    public class Solver
    {
        public double DecayRate { get; }
        public  double SmoothEpilson { get; }
        readonly List<Matrix> StepCache = new List<Matrix>();

        public Solver(double decayRate= 0.999, double smootheps=1e-8)
        {
            DecayRate = decayRate;
            SmoothEpilson = smootheps;
        }

        public Dictionary<string, double> Step(Matrix model, int step_size, int regc, double clipval)
        {
            var solverStats= new Dictionary<string,double>();
            var num_clipped = 0;
            var num_tot = 0;
            if (!IsInCache(StepCache, model))
            {
                StepCache.Add(new Matrix(model.NumberOfRows,model.NumberOfColumns,model.Id));
            }
            var s = StepCache.LastOrDefault();
            if (s != null)
            {
                for (int i = 0, n = model.Weights.Length; i < n; i++)
                {
                    // rmsprop adaptive learning rate
                    var mdwi =model.BackPropWeights[i];
                    s.Weights[i] = s.Weights[i] * DecayRate + (1.0 - DecayRate) * mdwi * mdwi;
                    // gradient clip
                    if (mdwi > clipval)
                    {
                        mdwi = clipval;
                        num_clipped++;
                    }
                    if (mdwi < -clipval)
                    {
                        mdwi = -clipval;
                        num_clipped++;
                    }
                    num_tot++;
                    // update (and regularize)
                    model.Weights[i] += -step_size * mdwi / Math.Sqrt(s.Weights[i] + SmoothEpilson) - regc * model.Weights[i];
                    model.BackPropWeights[i] = 0; // reset gradients for next iteration
                }
            }
            solverStats.Add("ratio_clipped", num_clipped * 1.0 / num_tot);
            return solverStats;
        }

        private bool IsInCache(List<Matrix>cache, Matrix model)
        {
            foreach (var matrix in cache)
            {
                if (matrix.Id == model.Id)
                {
                    return true;
                }
            }
            return false;
        }
    }
}
