using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Uppg_4;

//förutsägelser
//data går "ut"
public class PassengerPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Survived { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}
