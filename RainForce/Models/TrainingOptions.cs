using RainForce.Categories;

namespace RainForce.Models
{
    public class TrainingOptions
    {
        public UpdateMode UpdateMode { get; }
        public double Gamma { get; }
        public double Epsilon { get; set; }
        public double Alpha { get; set; }
        public int ExperienceAddEvery { get; set; }
        public int ExperienceSize { get; set; }
        public int LearningSteps { get; set; }
        public double ErrorClamp { get; set; }
        public int HiddenUnits { get; set; }
        public int ExperienceStart { get; set; }

        public TrainingOptions()
        {
            UpdateMode=UpdateMode.Qlearn;
            Gamma = 0.9;
        }
    }
}
