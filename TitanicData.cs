using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace Uppg_4;

//kolumner som modellen använder för att hitta mönster
//Data som går "in" i modellen (features)
public class TitanicData
{
    [LoadColumn(1)]
    //det värde som modellen ska lära sig att förutsäga
    //valt dessa som features då de verkar mest rimliga att kontrollera
    public bool Survived { get; set; }
    [LoadColumn(2)]
    public float Pclass { get; set; }
    [LoadColumn(4)]
    public string? Sex { get; set; }
    [LoadColumn(5)]
    public float Age { get; set; }
    [LoadColumn(6)]
    public float SibSp { get; set; } //antalet syskon/make/maka. familjeförhållanden kan ha påverkat chansen att överleva
    [LoadColumn(7)]
    public float Parch { get; set; } //föräldrar/barn
    [LoadColumn(9)]
    public float Fare { get; set; } //biljettpris. En person som betalade mer hade kanske, enligt modellen, högre chans för överlevnad
}