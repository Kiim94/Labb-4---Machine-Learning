using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Uppg_4;

class Program
{
    static void Main()
    {
        //skapa variabel för att kunna använda ML-funktioner
        var mlContext = new MLContext();

        //variabel för att hämta filen med datan
        IDataView data = mlContext.Data.LoadFromTextFile<TitanicData>(
            "Titanic-Dataset.csv", //filens namn
            hasHeader: true, //det finns headers/rubriker för kolumnerna
            separatorChar: ',', //separera kolumner med ,
            allowQuoting: true //gör att , inuti citat inte separeras (t.ex. ett namn som "Braun, James Brown" är en kolumn) 
            );

        //bygg pipeline
        //pipeline bestämmer hur rådata ska bearbetas och hur modell ska lära sig se mönster
        //ändrat så sex (male/female) istället vissas som ettor och nollor <- bättre för modellen
        //tomma rader för ålder: ersatt dem med mean (det tar samtliga existerande åldrar och hittar medelvärde)
        var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(
            outputColumnName: "SexEncoded",
            inputColumnName: "Sex")
            .Append(mlContext.Transforms.ReplaceMissingValues(
                outputColumnName: "AgeReplaced",
                inputColumnName: "Age",
                replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
            .Append(mlContext.Transforms.Concatenate(
                "Features",
                "Pclass",
                "SexEncoded",
                "AgeReplaced",
                "SibSp",
                "Parch",
                "Fare"
                ));

        //dela upp data i tränings- och testset
        //blir 80% och 20%
        var split = mlContext.Data.TrainTestSplit(
            data,
            testFraction: 0.2, seed: 1);

        //skapa en tränare för modellen
        //Survived är det värde som modellen ska lära sig att förutsäga
        //Features är de egenskaper moddelen ska använda för sina förutsägelser
        var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
            labelColumnName: nameof(TitanicData.Survived),
            featureColumnName: "Features");

        //träna modellen med en trainer på pipeline
        //pipeline bearbetar data och sedan används tränaren för att lära sig mönster
        var trainingPipeline = pipeline.Append(trainer);

        //träna modellen på träningsdata (80% av datasetet)
        //Fit() hanterar träningsdata, lär modellen samband mellan Features och Survived
        var trainModel = trainingPipeline.Fit(split.TrainSet);

        //testa den tränade modellen på testdata (20%)
        //modellen har inte tränats på denna data
        var testPrediction = trainModel.Transform(split.TestSet);

        //utvärdera modellen genom att jämföra på verkliga värden i testdata
        var testMetrics = mlContext.BinaryClassification.Evaluate(
            testPrediction, 
            labelColumnName: nameof(TitanicData.Survived));

        Console.WriteLine();
        Console.WriteLine("Utvärdering av 80/20 split av data:");
        Console.WriteLine($"Säkerhet: {testMetrics.Accuracy:P2}"); 
        //Andelen korrekta förutsägelser

        Console.WriteLine($"AUC: {testMetrics.AreaUnderRocCurve:P2}"); 
        //hur bra modellen skiljer mellan överlevande passagerare och de som inte överlevde

        Console.WriteLine($"F1 Score: {testMetrics.F1Score:P2}"); 
        //mått av precision och recall (precision: hur många av de förutsagda överleverna överlevde faktiskt
        //recall: hur stor andel av de faktiska överlevarna hittade modellen
        //nu närmare 1, desto bättre

        //träna ny modell på hela dataset
        //ska användas för att förutsäga överlevnad baserat på användarens input
        var model = trainingPipeline.Fit(data);

        //skapa en förutsägelse-maskin: predictionengine
        var predictionEngine = mlContext.Model.CreatePredictionEngine<TitanicData, PassengerPrediction>(model);


        //======================
        //tomma variabler som data från användaren ska sparas i
        float age;
        string sex;
        float fare;
        int pclass;
        int sibsp;
        int parch;

        //använt while för att användaren ska kunna skriva fel och sedan försöka igen
        Console.WriteLine();
        Console.WriteLine("===Your titanic passenger===");
        while (true)
        {
            Console.Write("Ange ålder: ");
            if(float.TryParse(Console.ReadLine(), out age) && age >= 0 && age <= 130)
            {
                break;
            }
            Console.WriteLine("Vänligen ange ett giltigt heltal (0-130)");
            Thread.Sleep(1500);
            Console.Clear();
        }
        while (true)
        {
            Console.Write("Ange kön (male/female): ");
            sex = Console.ReadLine()?.Trim().ToLower() ?? "";
            if (sex == "male" || sex == "female")
            {
                break;
            }
            Console.WriteLine("Vänligen ange antingen male eller female.");
            Thread.Sleep(1500);
            Console.Clear();
        }

        while (true)
        {
            Console.Write("Ange biljettpriset (t.ex 7,50 eller 20,50): ");
            if (float.TryParse(Console.ReadLine(), out fare) && fare >= 0)
            {
                break;
            }
            Console.WriteLine("Vänligen ange ett korrekt format");
            Thread.Sleep(1500);
            Console.Clear();
        }

        while (true)
        {
            Console.Write("Ange en klass (1-3): ");
            if (int.TryParse(Console.ReadLine(), out pclass) && pclass > 0 && pclass < 4)
            {
                break;
            }
            Console.WriteLine("Vänligen ange ett giltigt tal mellan 1 och 3");
            Thread.Sleep(1500);
            Console.Clear();
        }

        while (true)
        {
            Console.Write("Ange antal syskon/make/maka: ");
            if (int.TryParse(Console.ReadLine(), out sibsp) && sibsp >= 0)
            {
                break;
            }
            Console.WriteLine("Vänligen ange ett giltigt nummer.");
            Thread.Sleep(1500);
            Console.Clear();
        }

        while (true)
        {
            Console.Write("Ange antal föräldrar/barn: ");
            if (int.TryParse(Console.ReadLine(), out parch) && parch >= 0)
            {
                break;
            }
            Console.WriteLine("Vänligen ange ett giltigt nummer.");
            Thread.Sleep(1500);
            Console.Clear();
        }

        //spara data som användaren gett för en ny passagerare
        var passenger = new TitanicData
        {
            Age = age,
            Sex = sex,
            Fare = fare,
            Pclass = pclass,
            SibSp = sibsp,
            Parch = parch
        };

        //skicka in passageraren i engine för att få förutsägelse
        var prediction = predictionEngine.Predict(passenger);

        Console.WriteLine();
        Console.WriteLine("Resultat:");
        Console.WriteLine($"Modellen förutsäger att passageraren: {(prediction.Survived ? "Överlever" : "Överlever inte")}");
        Console.WriteLine($"Modellens uppskattade sannolikhet för överlevnad: {prediction.Probability:P2}"); // uppskattning för sannolikheten att personen överlever
        Console.WriteLine($"Score: {prediction.Score:F4}"); //positivt värde: överlever. negativt: överlever inte
    }
}
