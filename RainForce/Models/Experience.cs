namespace RainForce.Models
{
    public class Experience
    {
        public Matrix PreviousState { get; set; }
        public Matrix NextState { get; set; }
        public int PreviousAction, NextAction;
        public double Reward;
    }
}
