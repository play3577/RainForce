using System;
using System.Collections.Generic;
using RainForce.Models;

namespace RainForce.Utilities
{
    public static class Util
    {
        static bool GetFromMemory;
        static double CachedGaussRandom;
        private static int MatrixId { get; set; }

        public static int GetMatrixId()
        {
            MatrixId++;
            return MatrixId;
        }
        public static void Assert(bool condition, string message="")
        {
            if (!condition)
            {
                if (message == "")
                {
                    message = "Assertion Failled";
                }
                throw new Exception(message);
            }
        }

        public static double GaussRandom()
        {
            if (GetFromMemory)
            {
                GetFromMemory = false;
                return CachedGaussRandom;
            }
            var u = 2 * RandomMaxOne() - 1;
            var v = 2 * RandomMaxOne() - 1;
            var r = u * u + v * v;
            if (r == 0 || r > 1)
            {
                return GaussRandom();
            }
            var c = Math.Sqrt(-2 * Math.Log(r) / r);
            CachedGaussRandom = v * c;
            GetFromMemory = true;
            return u * c;
        }

        public static double RandomMaxOne(double minimum=0, double maximum=1)
        {
            Random random = new Random();
            return random.NextDouble() * (maximum - minimum) + minimum;
        }

        public static double Random(double a, double b)
        {
            return RandomMaxOne()* (b - a) + a;
        }

        public static int Random(int a, int b)
        {
            var t= Math.Floor(Random((double) a, b));
            return (int) t;
        }

        public static double Random(double mu, double std,double gauss)
        {
            return mu + gauss * std;
        }

        public static double[] ArrayOfZeros(int length=0)
        {
            if (length <= 0)
            {
                return new double[0];
            }
            else
            {
                var ck = new double[length];
                for (int i = 0; i < length; i++)
                {
                    ck[i] = 0;
                }
                return ck;
            }
        }

        public static double SigmoidHelper(double x)
        {
            // helper function for computing sigmoid
            return 1.0 / (1 + Math.Exp(-x));
        }

        public static void FillMatrixWithRandomGaussianNumbers(Matrix m,double mu, double std)
        {
            for (int i = 0, n = m.Weights.Length; i < n; i++)
            {
                m.Weights[i] = Random(mu, std,GaussRandom());
            }
        }

        public static Matrix Matrix(int rows,int columns)
        {
            return new Matrix(rows,columns);
        }

        public static Matrix Matrix(Matrix b)
        {
            var a= new Matrix(b.NumberOfRows,b.NumberOfColumns);
            a.Set(b.Weights);
            return a;
        }
        public static Network Net(Network network)
        {
            if (network.Key != string.Empty)
            {
                return network;
            }
            else
            {
                return null;
            }
        }

        public static void UpdateMatrix(Matrix m, double alpha)
        {
            var n = m.NumberOfRows * m.NumberOfColumns;
            for (var i = 0; i < n; i++)
            {
                if (m.BackPropWeights[i] > 0)
                {
                    m.Weights[i] += -alpha * m.BackPropWeights[i];
                    m.BackPropWeights[i] = 0;
                }
            }
        }
        public static void UpdateNetwork(Network m, double alpha)
        {
            foreach (var mMatrix in m.Matrices)
            {
                if (mMatrix != null)
                {
                    UpdateMatrix(mMatrix,alpha);
                }
            }

        }

        public static void FillBackPropWeightsWithConstant(Matrix m, int constant)
        {
            for (int i = 0, n = m.BackPropWeights.Length; i < n; i++)
            {
                m.BackPropWeights[i] = constant;
            }
        }

        public static Matrix Flatten(Matrix mat)
        {
            var n = mat.BackPropWeights.Length;
            var g= new Matrix(n,1);
            for (int i = 0, m =n; i < m; i++)
            {
                g.Weights[i] = mat.BackPropWeights[i];
            }
            return g;
        }

        public static int ActionFromWeights(double[] w)
        {
            // argmax of array w
            var maxv = w[0];
            var maxix = 0;
            for (int i = 1, n = w.Length; i < n; i++)
            {
                var v = w[i];
                if (v > maxv)
                {
                    maxix = i;
                    maxv = v;
                }
            }
            return maxix;
        }

        public static int SampleAction(double[] w)
        {
            // sample argmax from w, assuming w are
            // probabilities that sum to one
            var r = Random((double) 0, 1);
            var x = 0.0;
            var i = 0;
            while (true)
            {
                x += w[i];
                if (x > r)
                {
                    return i;
                }
                i++;
            }
        }

        public static int SampleWeightedAction(double[]p)
        {
            var r = RandomMaxOne();
            var c = 0.0;
            for (int i = 0, n = p.Length; i < n; i++)
            {
                c += p[i];
                if (c >= r)
                {
                    return i;
                }
            }
            Assert(false, "WTF");
            return 0;//quite sute will never get here
        }
        public static int SampleWeightedAction(List<double> p)
        {
            var r = RandomMaxOne();
            var c = 0.0;
            for (int i = 0, n = p.Count; i < n; i++)
            {
                c += p[i];
                if (c >= r)
                {
                    return i;
                }
            }
            Assert(false, "WTF");
            return 0;//quite sute will never get here
        }

        public static void FillArrayWithConstant(double[] arr, double con)
        {
            for (int i = 0, n = arr.Length; i < n; i++)
            {
                arr[i] = con;
            }
        }
        public static void FillArrayWithConstant(int[] arr, int con)
        {
            for (int i = 0, n = arr.Length; i < n; i++)
            {
                arr[i] = con;
            }
        }
    }
}
