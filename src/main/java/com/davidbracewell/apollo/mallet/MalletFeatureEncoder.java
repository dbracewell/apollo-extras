package com.davidbracewell.apollo.mallet;

import cc.mallet.types.Alphabet;
import com.davidbracewell.apollo.ml.Encoder;
import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.stream.MStream;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class MalletFeatureEncoder implements Encoder, Serializable {
   private static final long serialVersionUID = 1L;
   private final Alphabet alphabet;

   public MalletFeatureEncoder(Alphabet alphabet) {
      this.alphabet = alphabet;
   }

   @Override
   public Encoder createNew() {
      return new MalletFeatureEncoder(alphabet);
   }

   @Override
   public Object decode(double value) {
      return alphabet.lookupObject((int) value);
   }

   @Override
   public double encode(Object object) {
      return alphabet.lookupIndex(object);
   }

   @Override
   public void fit(Dataset<? extends Example> dataset) {

   }

   @Override
   public void fit(MStream<String> stream) {

   }

   @Override
   public void freeze() {
      alphabet.stopGrowth();
   }

   @Override
   public double get(Object object) {
      return alphabet.lookupIndex(object);
   }

   @Override
   public boolean isFrozen() {
      return alphabet.growthStopped();
   }

   @Override
   public int size() {
      return alphabet.size();
   }

   @Override
   public void unFreeze() {
      alphabet.startGrowth();
   }

   @Override
   public List<Object> values() {
      return Arrays.asList(alphabet.toArray());
   }

}// END OF MalletFeatureEncoder
