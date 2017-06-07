package com.davidbracewell.apollo.mallet.classification;

import cc.mallet.types.Alphabet;
import cc.mallet.types.Labeling;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.mallet.MalletFeatureEncoder;
import com.davidbracewell.apollo.mallet.MalletLabelEncoder;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classification;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierEvaluation;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.collection.map.Maps;

import java.util.Arrays;
import java.util.Random;

/**
 * @author David B. Bracewell
 */
public class MalletClassifier extends Classifier {
   private static final long serialVersionUID = 1L;
   cc.mallet.classify.Classifier model;

   /**
    * Instantiates a new Classifier.
    *
    * @param targetAlphabet  the target alphabet
    * @param featureAlphabet the feature alphabet
    * @param preprocessors   the preprocessors that the classifier will need apply at runtime
    */
   protected MalletClassifier(Alphabet targetAlphabet, Alphabet featureAlphabet, PreprocessorList<Instance> preprocessors) {
      super(new EncoderPair(new MalletLabelEncoder(targetAlphabet), new MalletFeatureEncoder(featureAlphabet)),
            preprocessors);
   }

   @Override
   public Classification classify(Vector vector) {
      Labeling labeling = model.classify(model.getInstancePipe()
                                              .instanceFrom(new cc.mallet.types.Instance(vector, "", null, null)))
                               .getLabeling();
      double[] result = new double[getLabelEncoder().size()];
      for (int i = 0; i < getLabelEncoder().size(); i++) {
         result[i] = labeling.value(i);
      }
      return createResult(result);
   }


   public static void main(String[] args) {
//      Resource url = Resources.from(
//         "https://raw.githubusercontent.com/sjwhitworth/golearn/master/examples/datasets/iris_headers.csv");
//      DenseCSVDataSource dataSource = new DenseCSVDataSource(url, true);
//      dataSource.setLabelName("Species");
      Dataset<Instance> dataset = Dataset.classification()
//                                         .source(dataSource)
                                         .source(Arrays.asList(
                                            Instance.create(Maps.map("love", 1.0, "hate", 0.0, "wife", 1.0), "Married"),
                                            Instance.create(Maps.map("love", 1.0, "hate", 0.0, "wife", 1.0), "Married"),
                                            Instance.create(Maps.map("love", 1.0, "hate", 0.0, "wife", 1.0), "Married"),
                                            Instance.create(Maps.map("love", 0.0, "hate", 1.0, "girlfriend", 1.0),
                                                            "NotMarried"),
                                            Instance.create(Maps.map("love", 0.0, "hate", 1.0, "girlfriend", 1.0),
                                                            "NotMarried"),
                                            Instance.create(Maps.map("love", 0.0, "hate", 1.0, "wife", 1.0), "Married")
                                                              ))
                                         .shuffle(new Random(1234));

      ClassifierLearner learner = new AdaBoostLearner();
      ClassifierEvaluation eval = new ClassifierEvaluation();
      eval.evaluate(learner.train(dataset), dataset);
      eval.output(System.out);
   }

}// END OF MalletClassifier
