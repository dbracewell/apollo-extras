package com.davidbracewell.apollo.mallet.topic;

import cc.mallet.pipe.SerialPipes;
import cc.mallet.topics.ParallelTopicModel;
import com.davidbracewell.apollo.affinity.Similarity;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.NoOptEncoder;
import com.davidbracewell.apollo.ml.NoOptLabelEncoder;
import com.davidbracewell.apollo.ml.clustering.flat.FlatClustering;

/**
 * @author David B. Bracewell
 */
public class MalletLDAModel extends FlatClustering {
   ParallelTopicModel topicModel;
   SerialPipes pipes;

   protected MalletLDAModel() {
      super(new EncoderPair(new NoOptLabelEncoder(), new NoOptEncoder()), Similarity.Cosine.asDistanceMeasure());
   }


   @Override
   public int hardCluster(Instance instance) {
      return 0;
   }

   @Override
   public double[] softCluster(Instance instance) {
      return new double[0];
   }


}//END OF MalletLDA
