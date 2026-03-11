//+------------------------------------------------------------------+
//|                                               FanoOctonion.mqh   |
//|                        Fano Plane Octonion Algebra Foundation     |
//|                                    QuantumChildren / DooDoo 2026 |
//+------------------------------------------------------------------+
#ifndef FANO_OCTONION_MQH
#define FANO_OCTONION_MQH

//--- Cayley multiplication table: sign of e_i * e_j
//    Derived from Fano plane triples:
//    (1,2,4) (2,3,5) (3,4,6) (4,5,7) (5,6,1) (6,7,2) (7,1,3)
const int CAYLEY_SIGN[8][8] =
{
   { 1,  1,  1,  1,  1,  1,  1,  1},  // e0
   { 1, -1,  1,  1, -1,  1, -1, -1},  // e1
   { 1, -1, -1,  1,  1, -1,  1, -1},  // e2
   { 1, -1, -1, -1,  1,  1, -1,  1},  // e3
   { 1,  1, -1, -1, -1,  1,  1, -1},  // e4
   { 1, -1,  1, -1, -1, -1,  1,  1},  // e5
   { 1,  1, -1,  1, -1, -1, -1,  1},  // e6
   { 1,  1,  1, -1,  1, -1, -1, -1}   // e7
};

//--- Cayley multiplication table: result index k where e_i * e_j = sign * e_k
const int CAYLEY_IDX[8][8] =
{
   {0, 1, 2, 3, 4, 5, 6, 7},  // e0
   {1, 0, 4, 7, 2, 6, 5, 3},  // e1
   {2, 4, 0, 5, 1, 3, 7, 6},  // e2
   {3, 7, 5, 0, 6, 2, 4, 1},  // e3
   {4, 2, 1, 6, 0, 7, 3, 5},  // e4
   {5, 6, 3, 2, 7, 0, 1, 4},  // e5
   {6, 5, 7, 4, 3, 1, 0, 2},  // e6
   {7, 3, 6, 1, 5, 4, 2, 0}   // e7
};

//+------------------------------------------------------------------+
//| Octonion — 8-dimensional hypercomplex number                     |
//+------------------------------------------------------------------+
struct Octonion
{
   double v[8]; // components e0..e7

   //--- Initialize to zero
   void Zero()
   {
      for(int i = 0; i < 8; i++)
         v[i] = 0.0;
   }

   //--- Multiplicative identity (1, 0, 0, 0, 0, 0, 0, 0)
   void Unit()
   {
      Zero();
      v[0] = 1.0;
   }

   //--- Set all eight components
   void Set(double e0, double e1, double e2, double e3,
            double e4, double e5, double e6, double e7)
   {
      v[0] = e0; v[1] = e1; v[2] = e2; v[3] = e3;
      v[4] = e4; v[5] = e5; v[6] = e6; v[7] = e7;
   }

   //--- Unit basis vector: e_idx = 1, all others = 0
   void SetBasis(int idx)
   {
      Zero();
      if(idx >= 0 && idx < 8)
         v[idx] = 1.0;
   }

   //--- Euclidean norm: ||q|| = sqrt(sum v[i]^2)
   double Norm()
   {
      return MathSqrt(NormSq());
   }

   //--- Squared norm: sum v[i]^2
   double NormSq()
   {
      double s = 0.0;
      for(int i = 0; i < 8; i++)
         s += v[i] * v[i];
      return s;
   }

   //--- Real (scalar) part
   double Real()
   {
      return v[0];
   }

   //--- Mean of imaginary components v[1..7]
   void ImagMean(double &result)
   {
      double s = 0.0;
      for(int i = 1; i < 8; i++)
         s += v[i];
      result = s / 7.0;
   }

   //--- Conjugate: q* = (v[0], -v[1], ..., -v[7])
   void Conjugate(Octonion &result)
   {
      result.v[0] = v[0];
      for(int i = 1; i < 8; i++)
         result.v[i] = -v[i];
   }

   //--- Component-wise addition
   void Add(const Octonion &b, Octonion &result)
   {
      for(int i = 0; i < 8; i++)
         result.v[i] = v[i] + b.v[i];
   }

   //--- Component-wise subtraction
   void Sub(const Octonion &b, Octonion &result)
   {
      for(int i = 0; i < 8; i++)
         result.v[i] = v[i] - b.v[i];
   }

   //--- Scalar multiplication
   void Scale(double s, Octonion &result)
   {
      for(int i = 0; i < 8; i++)
         result.v[i] = v[i] * s;
   }

   //--- Octonion product using Cayley table (non-associative, non-commutative)
   void Multiply(const Octonion &b, Octonion &result)
   {
      for(int k = 0; k < 8; k++)
         result.v[k] = 0.0;

      for(int i = 0; i < 8; i++)
      {
         for(int j = 0; j < 8; j++)
         {
            int k = CAYLEY_IDX[i][j];
            int s = CAYLEY_SIGN[i][j];
            result.v[k] += s * v[i] * b.v[j];
         }
      }
   }
};

#endif // FANO_OCTONION_MQH
