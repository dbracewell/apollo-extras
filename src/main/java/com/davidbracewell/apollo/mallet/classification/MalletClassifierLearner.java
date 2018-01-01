package com.davidbracewell.apollo.mallet.classification;

import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.pipe.*;
import cc.mallet.types.InstanceList;
import com.davidbracewell.apollo.mallet.VectorToTokensPipe;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.data.Dataset;

import java.util.Arrays;

/**
 * @author David B. Bracewell
 */
public abstract class MalletClassifierLearner extends ClassifierLearner {
   private static final long serialVersionUID = 1L;

   protected SerialPipes createPipe() {
      return new SerialPipes(Arrays.asList(new Target2Label(),
                                           new VectorToTokensPipe(getEncoderPair().getFeatureEncoder()),
                                           new TokenSequence2FeatureSequence(),
                                           new FeatureSequence2FeatureVector()));
   }

   protected abstract ClassifierTrainer<?> getTrainer();

   @Override
   protected void resetLearnerParameters() {

   }

   @Override
   protected final Classifier trainImpl(Dataset<Instance> dataset) {
      Pipe pipe = createPipe();
      InstanceList trainingData = new InstanceList(pipe);
      dataset.asVectors()
             .forEach(i -> {
                String lbl = dataset.getEncoderPair().decodeLabel(i.getLabel()).toString();
                trainingData.addThruPipe(new cc.mallet.types.Instance(i, lbl, null, null));
             });
      ClassifierTrainer<?> trainer = getTrainer();
      MalletClassifier model = new MalletClassifier(this);
      model.model = trainer.train(trainingData);
      return model;
   }


}// END OF MalletClassifierLearner
