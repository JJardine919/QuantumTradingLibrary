//+------------------------------------------------------------------+
//|                                          FanoDecomposition.mqh   |
//|              Jordan-Shadow Decomposition of Octonion Products    |
//|                                 QuantumChildren / DooDoo 2026    |
//+------------------------------------------------------------------+
#ifndef FANO_DECOMPOSITION_MQH
#define FANO_DECOMPOSITION_MQH

#include <Fano\FanoOctonion.mqh>

//+------------------------------------------------------------------+
//| FanoDecomp — Jordan-Shadow regime decomposition                  |
//|                                                                  |
//| Given signal octonion A and momentum octonion B:                 |
//|   Jordan     = (AB + BA) / 2   symmetric  — CONSENSUS            |
//|   Commutator = (AB - BA) / 2   anti-sym   — CONFLICT             |
//|   Associator = Jordan * Commutator         — CHAOS/TRANSITION    |
//+------------------------------------------------------------------+
struct FanoDecomp
{
   Octonion A;                  // signal octonion (1, s1..s7)
   Octonion B;                  // momentum octonion (1, ds1..ds7)
   Octonion AB;                 // A * B
   Octonion BA;                 // B * A
   Octonion Jordan;             // (AB + BA) / 2 — consensus
   Octonion Commutator;         // (AB - BA) / 2 — conflict
   Octonion Associator;         // Jordan * Commutator — chaos

   double   jordan_strength;    // |mean(Jordan.v[1..7])|
   double   commutator_strength;// |mean(Commutator.v[1..7])|
   double   chaos_level;        // Norm(Associator)
   int      jordan_direction;   // +1 BUY, -1 SELL, 0 FLAT

   //--- Compute full decomposition from 7 signals and 7 deltas
   void Compute(const double &signals[], const double &dsignals[])
   {
      // Build signal octonion: e0=1, e1..e7 = signals
      A.v[0] = 1.0;
      for(int i = 0; i < 7; i++)
         A.v[i + 1] = signals[i];

      // Build momentum octonion: e0=1, e1..e7 = dsignals
      B.v[0] = 1.0;
      for(int i = 0; i < 7; i++)
         B.v[i + 1] = dsignals[i];

      // Compute products (non-commutative)
      A.Multiply(B, AB);
      B.Multiply(A, BA);

      // Jordan = (AB + BA) / 2
      Octonion sum;
      AB.Add(BA, sum);
      sum.Scale(0.5, Jordan);

      // Commutator = (AB - BA) / 2
      Octonion diff;
      AB.Sub(BA, diff);
      diff.Scale(0.5, Commutator);

      // Associator = Jordan * Commutator
      Jordan.Multiply(Commutator, Associator);

      // Scalar summaries
      double j_mean = 0.0;
      Jordan.ImagMean(j_mean);
      jordan_strength = MathAbs(j_mean);

      double c_mean = 0.0;
      Commutator.ImagMean(c_mean);
      commutator_strength = MathAbs(c_mean);

      chaos_level = Associator.Norm();

      // Direction from Jordan consensus
      if(j_mean > 0.001)
         jordan_direction = 1;        // BUY
      else if(j_mean < -0.001)
         jordan_direction = -1;       // SELL
      else
         jordan_direction = 0;        // FLAT
   }
};

#endif // FANO_DECOMPOSITION_MQH
