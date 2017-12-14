using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using RainForce.Agents;
using RainForce.Models;

namespace Example
{
    class Program
    {
        static void Main(string[] args)
        {
            var rnd = new Random();
            int max = 10;
            int min = 1;
            int nextPrint=0;
            double total=0,correct=0;
            var state = new[] {rnd.Next(min,max), rnd.Next(min, max), rnd.Next(min, max), rnd.Next(min, max) };
            var opt = new TrainingOptions
            {
                Alpha = 0.001,
                Epsilon = 0.5,
                ErrorClamp = 0.002,
                ExperienceAddEvery = 10,
                ExperienceSize = 1000,
                ExperienceStart = 0,
                HiddenUnits = 5,
                LearningSteps = 400
            };
            //we take 4 states i.e random numbers between 1 and 10
            //we have 2 actions 1 if average of set is >5 and 0 if otherwise
            //we reward agent with 1 for every correct and -1 otherwise
            var agent= new DQNAgent(opt,state.Length,2);

            //how to properly use the DPAgent
            var agent2= new MyDPAgent();
            agent2.Reset(state.Length,2);

            while (total < 50000)
            {
                state = new[] {rnd.Next(min, max), rnd.Next(min, max), rnd.Next(min, max), rnd.Next(min, max)};
                var action = agent.Act(state);
                if (state.Average() > 5 && action == 1)
                {
                    agent.Learn(1);
                    correct++;
                }
                else if (state.Average() <= 5 && action == 0)
                {
                    agent.Learn(1);
                    correct++;
                }
                else
                {
                    agent.Learn(-1);
                }
                total++;
                //nextPrint++;
                if (total >= nextPrint)
                {
                    Console.WriteLine("Score: " + (correct / total).ToString("P")+"Epoch: "+nextPrint);
                    nextPrint += 1000;
                }
            }
           // Console.WriteLine("Score: " + (correct / total).ToString("P"));
            Console.WriteLine("End");
            File.AppendAllText(AppDomain.CurrentDomain.BaseDirectory + "DNQ.trr", agent.AgentToJson());
            Console.ReadKey();
        }
    }

    public class MyDPAgent : DPAgent
    {
        protected override int[] GetAllowedActions(int s)
        {
            return new[] {0, 1, 2, 3};
        }

        protected override int Reward(int s, int a, int d)
        {
            return s * a * d;
        }

        protected override int NextStateDistribution(int s, int a)
        {
            return s * a;
        }
    }
}
