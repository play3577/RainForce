using System;
using Newtonsoft.Json;

namespace RainForce.Models
{
    public class Network
    {
        public string Key { get; private set; }
        public Matrix[] Matrices { get; private set; }
        //public Matrix Matrix { get; set; }
        //public Matrix W1 { get; set; }
        //public Matrix B1 { get; set; }
        //public Matrix W2 { get; set; }
        //public Matrix B2 { get; set; }

        public Network(int matrixCount,string key="")
        {
            Matrices=new Matrix[matrixCount];
            Key=key;
        }

        public string ToJson()
        {
            return JsonConvert.SerializeObject(this);
        }

        public static Network FromJson(string json)
        {
            return JsonConvert.DeserializeObject<Network>(json);
        }

    }
}