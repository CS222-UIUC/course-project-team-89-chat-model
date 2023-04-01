
//this is a class that is used to produce machine learning code and write to a file
class MLFileWriter {
    public static string path;
    private static string code;
    public MLFileWriter(List<string> specifications, string p = "file.py") {
        path = p;
        code = generateCode(specifications);
        WriteToFile(code);
    }

    public string getCode() {
        return code;
    }

    //this method generates the code for the machine learning file
    private string generateCode(List<string> specifications) {
        string code = "";
        
        //read from age_gender_classifier.py in src folder
        var file_path = "../src/age_gender_classifier.py";
        using (StreamReader sr = new StreamReader(file_path)) {
            code = sr.ReadToEnd();
        }

        var deft = new List<string> () {"age_gender.csv", "img_name", "gender, age, ethnicity", "sgd", "0.25", "0.8", "110", "20", "16", "SameerKomoravoluBW.jpg"};
        if (specifications.Count < deft.Count) {
            specifications = deft;
        }
        for (int i = 0; i < specifications.Count; i++) {
            code = code.Replace("spc" + i, specifications[i]);
        }
        return code;
    }

    private static void WriteToFile(string code) {
        using (StreamWriter sw = new StreamWriter(path)) {
            sw.WriteLine(code);
            Console.WriteLine("Successfully wrote to file");
        }
    }
}
