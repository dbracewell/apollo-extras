package com.davidbracewell.apollo.mallet.topic;

import cc.mallet.pipe.SerialPipes;
import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.topics.TopicInferencer;
import cc.mallet.types.Alphabet;
import cc.mallet.types.IDSorter;
import cc.mallet.types.InstanceList;
import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.linalg.SparseVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.clustering.topic.TopicModel;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.collection.counter.Counters;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.TreeSet;

/**
 * The type Mallet lda model.
 *
 * @author David B. Bracewell
 */
public class MalletLDAModel extends TopicModel {
   private static final long serialVersionUID = 1L;
   /**
    * The Topic model.
    */
   ParallelTopicModel topicModel;
   /**
    * The Pipes.
    */
   SerialPipes pipes;
   /**
    * The Inferencer.
    */
   transient TopicInferencer inferencer;

   /**
    * Instantiates a new Mallet lda model.
    *
    * @param encoderPair     the encoder pair
    * @param distanceMeasure the distance measure
    */
   MalletLDAModel(EncoderPair encoderPair, DistanceMeasure distanceMeasure) {
      super(encoderPair, distanceMeasure);
   }

   private TopicInferencer getInferencer() {
      if (inferencer == null) {
         synchronized (this) {
            if (inferencer == null) {
               inferencer = topicModel.getInferencer();
               inferencer.setRandomSeed(1234);
            }
         }
      }
      return inferencer;
   }

   /**
    * Gets topic alpha.
    *
    * @param topicId the topic id
    * @return the topic alpha
    */
   public double getTopicAlpha(int topicId) {
      return topicModel.alpha[topicId];
   }

   @Override
   public double[] getTopicDistribution(String feature) {
      final Alphabet alphabet = pipes.getDataAlphabet();
      int index = alphabet.lookupIndex(feature, false);
      if (index == -1) {
         return new double[getK()];
      }
      double[] dist = new double[getK()];
      double[][] termScores = topicModel.getTopicWords(true, true);
      for (int i = 0; i < getK(); i++) {
         dist[i] = termScores[i][index];
      }
      return dist;
   }

   @Override
   public Vector getTopicVector(int topic) {
      Counter<String> topicCtr = getTopicWords(topic);
      Vector v = SparseVector.zeros(getFeatureEncoder().size());
      topicCtr.entries().forEach(e -> v.set(getFeatureEncoder().index(e.getKey()), e.getValue()));
      return v;
   }

   @Override
   public Counter<String> getTopicWords(int topic) {
      final Alphabet alphabet = pipes.getDataAlphabet();
      final ArrayList<TreeSet<IDSorter>> topicWords = topicModel.getSortedWords();
      double[][] termScores = topicModel.getTopicWords(true, true);
      Iterator iterator = topicWords.get(topic).iterator();
      IDSorter info;
      Counter<String> topicWordScores = Counters.newCounter();
      while (iterator.hasNext()) {
         info = (IDSorter) iterator.next();
         topicWordScores.set(alphabet.lookupObject(info.getID()).toString(), termScores[topic][info.getID()]);
      }
      return topicWordScores;
   }

   @Override
   public double[] softCluster(@NonNull Instance instance) {
      InstanceList instances = new InstanceList(pipes);
      instances.addThruPipe(new cc.mallet.types.Instance(instance.toVector(getEncoderPair()), "", null, null));
      return getInferencer().getSampledDistribution(instances.get(0), 800, 5, 100);
   }


}//END OF MalletLDA
