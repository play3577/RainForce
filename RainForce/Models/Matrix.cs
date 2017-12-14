using Newtonsoft.Json;
using RainForce.Utilities;

namespace RainForce.Models
{
    public class Matrix
    {
        public int NumberOfRows { get; private set; }
        public int NumberOfColumns { get; }
        public double[] Weights { get; set; }
        public double[] BackPropWeights { get; }
        public int Id { get; }

        public Matrix(int rows, int columns)
        {
            Id = Util.GetMatrixId();
            NumberOfRows = rows;
            NumberOfColumns = columns;
            Weights = Util.ArrayOfZeros(rows * columns);
            BackPropWeights = Util.ArrayOfZeros(rows * columns);
        }
        public Matrix(int rows, int columns,int id)
        {
            Id = id;
            NumberOfRows = rows;
            NumberOfColumns = columns;
            Weights = Util.ArrayOfZeros(rows * columns);
            BackPropWeights = Util.ArrayOfZeros(rows * columns);
        }

        public Matrix()
        {
            Id = Util.GetMatrixId();
        }

        public double Get(int row, int col)
        {
            var ix = (NumberOfColumns * row) + col;
            Util.Assert(ix >= 0 && ix < Weights.Length);
            return Weights[ix];
        }
        public void Set(int row, int col,double value)
        {
            var ix = (NumberOfColumns * row) + col;
            Util.Assert(ix >= 0 && ix < Weights.Length);
            Weights[ix] = value;
        }

        public void Set(double[] arr)
        {
            NumberOfRows = arr.Length;
            Weights = arr;
        }
        public void Set(int[] arr)
        {
            NumberOfRows = arr.Length;
            Weights = new double[NumberOfRows];
            for (int i = 0; i < NumberOfRows; i++)
            {
                Weights[i] = arr[i];
            }
        }

        public void Set(Matrix m, double i)
        {
            for (var q = 0; q < m.Weights.Length; q++)
            {
                int f = (int) (NumberOfColumns * q + i);
                Weights[f] = m.Weights[q];
            }
        }

        public string ToJson()
        {
            return JsonConvert.SerializeObject(this);
        }

        public static Matrix FromJson(string json)
        {
            return JsonConvert.DeserializeObject<Matrix>(json);
        }
    }
}