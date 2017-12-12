using System.Collections.Generic;
using RainForce.Models;

namespace RainForce.Utilities
{
    public class Solver
    {
        public double DecayRate { get; set; }
        private double SmoothEpilson = 1e-8;
        List<Matrix> StepCache = new List<Matrix>();
    }
}
