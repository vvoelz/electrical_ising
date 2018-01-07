// need to choose /MT for runtime library (Code generation) and Sequential option for Intel MKL

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <malloc.h>
#include <stdlib.h>
#include <float.h>
#include <mkl.h>
#include <mkl_vsl.h>

#define SQR(x)              ((x)*(x))
#define MAX(x,y)            ((x)>(y)?(x):(y))
#define MIN(x,y)            ((x)<(y)?(x):(y))
#define ABS(x)              ((x) > 0.0 ? (x) : -(x))
#define ABSMAX(x,y)         (ABS(x)>ABS(y)?ABS(x):ABS(y))
#define SIGN(a,b)           ((b) >= 0.0 ? ABS(a) : -ABS(a))
#define SWAP(type, a, b)    { type temp; temp = a; a = b; b = temp; }

//#define METHOD  VSL_RNG_METHOD_UNIFORM_STD
#define METHOD  VSL_RNG_METHOD_UNIFORM_STD_ACCURATE
//  ACCURACY FLAG FOR DISTRIBUTION GENERATORS
//  This flag defines mode of random number generation.
//  If accuracy mode is set distribution generators will produce
//  numbers lying exactly within definitional domain for all values
//  of distribution parameters. In this case slight performance
//  degradation is expected. By default accuracy mode is switched off
//  admitting random numbers to be out of the definitional domain for
//  specific values of distribution parameters.

#define _CRT_RAND_S 

typedef double REAL;
typedef int INTEGER;
typedef char CHARACTER;
typedef double* VECTOR;
typedef double** MATRIX;
typedef int* IVECTOR;
typedef int** IMATRIX;

typedef struct IntRange
{
        INTEGER m1;
        INTEGER m2;
        INTEGER n1;
        INTEGER n2;
} INDEX;

// subroutines
void InitState (IMATRIX st, INDEX grd, REAL prob, long *idum, VSLStreamStatePtr stream);
INTEGER ActiveCells (IMATRIX st, INDEX grd);
INTEGER NumActive (INTEGER i, INTEGER j, IMATRIX st, INDEX grd);
INTEGER WangLandau(IMATRIX st, MATRIX log_g, VECTOR q, VECTOR E, REAL Fo, REAL F_final, INTEGER Hcheck, INTEGER Ncells, INDEX grd, INDEX div, long *idum, VSLStreamStatePtr stream);
INTEGER Ising_MA (IMATRIX st, MATRIX P, MATRIX Pbias, MATRIX A_2, MATRIX A_1, MATRIX A0, MATRIX A1, MATRIX A2, MATRIX B_2, MATRIX B_1, MATRIX B0, MATRIX B1, MATRIX B2,
					INTEGER npts, INDEX grd, INDEX div, long *idum, VSLStreamStatePtr stream);
INTEGER Ising_MC (IMATRIX st, INTEGER npts, VECTOR tq, VECTOR aq, VECTOR bq, VECTOR E, MATRIX tqE, REAL q1, REAL q2,
                  INTEGER *acount, REAL *amfpt, INTEGER *bcount, REAL *bmfpt, INDEX grd, INDEX div, long *idum, VSLStreamStatePtr stream);
INTEGER Master_MC (CHARACTER* filename, INTEGER *st, MATRIX Q, IVECTOR na, INTEGER npts, REAL q1, REAL q2,
                  INTEGER *acount, REAL *amfpt, INTEGER *bcount, REAL *bmfpt, INDEX bnd, long *idum, VSLStreamStatePtr stream);
void SavePar(void);
INTEGER ReadPar(void);
void SaveRateModel(CHARACTER* filename, VECTOR q, VECTOR aq, VECTOR bq, VECTOR W, INDEX sts);
void SaveLandscape(CHARACTER* filename, VECTOR q, VECTOR E, MATRIX M, INDEX indx);
INTEGER ReadLandscape(CHARACTER* filename, MATRIX M, INDEX indx);
void SaveIntLandscape(CHARACTER* filename, VECTOR q, VECTOR E, IMATRIX M, INDEX indx);
void SaveMatrix(CHARACTER* filename, VECTOR q, VECTOR E, MATRIX M, INDEX indx);
void SaveVector(CHARACTER* filename, VECTOR q, VECTOR vv, INDEX indx);
INTEGER ReadVector(CHARACTER* filename, VECTOR vv, INDEX indx);
void SaveResults(CHARACTER* filename, INTEGER Ncells, INTEGER nst, INTEGER m, REAL Fo, INTEGER Hcheck, INTEGER na1, REAL na1q, INTEGER nab, REAL nabq, INTEGER na2,
	REAL na2q, INTEGER st1, REAL st1q, INTEGER st2, REAL st2q, INTEGER acount, INTEGER bcount, REAL amfpt, REAL bmfpt, INTEGER acount2D, INTEGER bcount2D,
	REAL amfpt2D, REAL bmfpt2D, INTEGER acount1D, INTEGER bcount1D, REAL amfpt1D, REAL bmfpt1D, REAL K, REAL a2D, REAL a1D, REAL a_ev_1D, REAL a_ev_2D);
void bandec(MATRIX a, INTEGER m1, INTEGER m2, MATRIX al, IVECTOR indx, INDEX sts);
void banbks(MATRIX a, INTEGER m1, INTEGER m2, MATRIX al, IVECTOR indx, VECTOR b, INDEX sts);
void lubksb(MATRIX a, IVECTOR indx, VECTOR b, INDEX sts);
INTEGER ludcmp(MATRIX a, IVECTOR indx, INDEX sts);
INTEGER EigenRate(VECTOR ev, MATRIX Q, INDEX bnd);
void Calc_Rate_Pair(MATRIX Q, MATRIX Qsym, VECTOR aq, VECTOR bq, MATRIX A1, INTEGER na1, INTEGER nE1, MATRIX B2, INTEGER na2, INTEGER nE2,
                     VECTOR PqE, IMATRIX state, INTEGER st, VECTOR stq, VECTOR stE,INTEGER m);
VECTOR MakeVector (INDEX indx);
void FreeVector (VECTOR V, INDEX indx);
IVECTOR MakeIntVector (INDEX indx);
void FreeIntVector (IVECTOR iV, INDEX indx);
void FreeIntMatrix(IMATRIX M, INDEX indx);
IMATRIX MakeIntMatrix (INDEX indx);
void FreeMatrix(MATRIX M, INDEX indx);
MATRIX MakeMatrix (INDEX indx);
long SeedRandomNumber (void);
REAL RandomNumber(long *idum);
unsigned GetSeed(void);
void EigenSystemXDE (MATRIX lEV, MATRIX rEV, VECTOR ev, MATRIX Ap, VECTOR Pe, INDEX tdi);
void EigSort(INDEX sts, VECTOR ev, MATRIX EV);
void EigSort_(VECTOR ev, INDEX sts);
void tqli(VECTOR d, VECTOR e, INDEX sts, MATRIX Z);
REAL pythag(REAL a, REAL b);
//extern errno_t rand_s(unsigned int* randomValue);

// global variables with default values
INTEGER	mode		= 0,
		rng_mode	= 0,
		npts		= 4000,
		ntr			= 9,
		grid_length	= 20,
		cell_length	= 20,
		grid_margin = 40,
		plot_xo		= 600,
		plot_yo		= 200,
		filter_div	= 100;

REAL	V		= 0.0,
		dV		= 0.0,
		TdegC	= 22.0,
		dTdegC	= 0.0,
		qmax	= 1.0,
		fs		= 100.0,
		fc		= 10.0,
		fi		= 1e6,
		ao		= 50000.0,
		logra	= 0.0,
		Ko		= 1.0,
		logrK	= 0.0,
		x		= 0.5,
		dx		= 0.0,
		eps		= 24.0,
		deps	= 0.0,
		w1		= 0.0,
		w2		= 1.0;

INTEGER rng = VSL_BRNG_SFMT19937,
		gi=1,
		gj=1;

REAL	gsc[5] = {0.0,0.0,0.0,0.0,1.0},
		gmax = 100.0,
		Vrev = -60.0,
		F_final = 1.0e-6,
		trunc_height = 0,
		Fmax = 150;

CHARACTER BiasFile[80];

int main()
{
    INDEX   grd,div,sts,bnd,tst,fpt,tdi;

    INTEGER i,j,tr,ndiv,exit=0,Ncells,na,nE,acount,bcount,acount1D,bcount1D,acount2D,bcount2D;
	INTEGER ii,jj;

    REAL prob,Z,lnZ,delq,T,kT,beta,tmp,amfpt,bmfpt,amfpt1D,bmfpt1D,amfpt2D,bmfpt2D;
	REAL lngmin;
	REAL Fo = 1.0;
	INTEGER Hcheck = 1000;
	MATRIX lng_WL,Pbias;

    FILE *fp;
	MATRIX tqE;
	VECTOR tq,aq,bq;

	MATRIX P,lng,A_2,A_1,A0,A1,A2,B_2,B_1,B0,B1,B2;
	VECTOR q,E,Wq;
    IMATRIX st;

    REAL Fmin,a1D,a2D,na1q,nabq,na2q,st1q,st2q,K,a_ev_1D,a_ev_2D;
    INTEGER m,nst,st1,st2,na1,nab,na2;
    MATRIX F,Q,Qsym,Qc,Ql,lEV,rEV;
    VECTOR Pq,PqE,stq,stE,u,uu,neg1,ev;
    IMATRIX state,Ones;
	IVECTOR Na,NE,indx,iu;

	long idum;
	VSLStreamStatePtr stream;
	unsigned seed;

	ReadPar();
	SavePar();

    // calculate dependencies

	ndiv = SQR(grid_length);
	delq = qmax/ndiv;
	T = TdegC+273.15;
    kT = 0.086174*T;
    beta = (kT == 0.0)?0.0:1.0/kT;

	// determine random number generator
	switch (rng_mode)
	{
		case 0:
			rng = VSL_BRNG_SFMT19937;
			break;
		case 1:
			rng = VSL_BRNG_MCG31;
			break;
		case 2:
			rng = VSL_BRNG_MRG32K3A;
			break;
		default:
			rng = VSL_BRNG_SFMT19937;
	}
	
	// define index limits

    grd.m1 = 1;
    grd.m2 = grid_length;
    grd.n1 = 1;
    grd.n2 = grid_length;

    div.m1 = 0;
    div.m2 = ndiv;
    div.n1 = 0;
    div.n2 = ndiv;

    // Allocate memory for vectors and matrices

    st		    = MakeIntMatrix(grd);
    state	    = MakeIntMatrix(div);
    Ones	    = MakeIntMatrix(div);
	u		    = MakeVector(div);
	q		    = MakeVector(div);
    E		    = MakeVector(div);
    tq		    = MakeVector(div);
    aq		    = MakeVector(div);
    bq		    = MakeVector(div);
    Wq		    = MakeVector(div);
    tqE		    = MakeMatrix(div);
    lng_WL	    = MakeMatrix(div);
    Pbias	    = MakeMatrix(div);
 	P		    = MakeMatrix(div);
    Pq          = MakeVector(div);
 	lng		    = MakeMatrix(div);
 	A_2		    = MakeMatrix(div);
 	A_1		    = MakeMatrix(div);
 	A0		    = MakeMatrix(div);
 	A1		    = MakeMatrix(div);
 	A2		    = MakeMatrix(div);
 	B_2		    = MakeMatrix(div);
 	B_1		    = MakeMatrix(div);
 	B0		    = MakeMatrix(div);
 	B1		    = MakeMatrix(div);
 	B2		    = MakeMatrix(div);
 	F		    = MakeMatrix(div);

	// values of q and E

	for(i=div.n1;i<=div.n2;i++)
	{
	    u[i] = i;               // unit vector
		q[i] = i*delq;
		E[i] = 2*i*eps;
	}

	//set seed for all simulations

	idum = SeedRandomNumber();
	seed = GetSeed();
	vslNewStream(&stream, rng, seed);

    //*************************************************************************
	//********************* Ones-matrix Calculation ****************************
	//*************************************************************************

	ii = ndiv/2;
	Ones[div.m1][div.n1] = Ones[div.m2][div.n1] = 1;

	for(i=div.m1;i<=ii;i++)		//na
		for(j=div.n1;j<=2*i;j++)		//nE
			Ones[ndiv-i][j] = Ones[i][j] = 1;

	Ones[ii][div.n2-1] = Ones[ii][div.n2-2] = Ones[ii-1][div.n2-3] = Ones[ii+1][div.n2-3] = 0;

	jj = 0;
	for(j=div.n1;j<=grid_length;j++)		//na
	{
		jj += (INTEGER) ceil(j/2);
		for(i=jj+1;i<=div.m2-jj-1;i++)		//nE
			Ones[i][j] = 0;
	}

	for(i=ii;i>=jj+1;i-=grid_length)
	{
		Ones[i][grid_length] = 1;
		Ones[i][grid_length+1] = 0;
		Ones[ndiv-i][grid_length] = 1;
		Ones[ndiv-i][grid_length+1] = 0;
	}

	if ((grid_length-2)%4 == 0 && grid_length > 2)
	{
		i += grid_length;
		Ones[i][grid_length+1] = 1;
		Ones[ndiv-i][grid_length+1] = 1;
	}

	Ncells = 0;
	for(i=div.m1;i<=div.m2;i++)		//na
		for(j=div.n1;j<=div.n2;j++)		//nE
			Ncells += Ones[i][j];

	SaveIntLandscape("qE_Ones.txt", u, u, Ones, div);

	//*************************************************************************
	//********************* Wang-Landau Simulation ****************************
	//*************************************************************************

	if (abs(mode) <= 1)		// skip if abs(mode) > 1
	{
		printf("\n\n** Wang-Landau simulation **\n");

		// pick initial conditions

		prob = 0.5;
		InitState(st, grd, prob, &idum, stream);

		// perform MC simulation
		exit = WangLandau(st, lng_WL, u, u, Fo, F_final, Hcheck, Ncells, grd, div, &idum, stream);

		lngmin = 0.0;
		for (ii = div.m1; ii <= div.m2; ii++)
			for (jj = div.n1; jj <= div.n2; jj++)
				lngmin += lng_WL[ii][jj];

		for (ii = div.m1; ii <= div.m2; ii++)
			for (jj = div.n1; jj <= div.n2; jj++)
				if (lng_WL[ii][jj] != 0.0 && lngmin > lng_WL[ii][jj])
					lngmin = lng_WL[ii][jj];

		for (ii = div.m1; ii <= div.m2; ii++)
			for (jj = div.n1; jj <= div.n2; jj++)
				if (lng_WL[ii][jj] != 0.0)
					lng_WL[ii][jj] -= lngmin;

		lng_WL[div.m1][div.n1] = 1e-10;
		lng_WL[div.m2][div.n1] = 1e-10;

		SaveLandscape("qE_lng_WL.txt", u, u, lng_WL, div);

		for (ii = div.n1; ii <= div.n2; ii++)
			for (jj = div.n1; jj <= div.n2; jj++)
				if (lng_WL[ii][jj] != 0.0)
					Pbias[ii][jj] = exp(lng_WL[ii][jj]);

		SaveLandscape("qE_Pbias.txt", u, u, Pbias, div);

		if (mode < 0)
		{
			printf("\n\nCompleted Wang-Landau simulation. Press any key to exit.\n");
			goto END_PROGRAM_1;
		}
	}
	else			// read Pbias from file, abort if file does not exist
	{
		printf("\n\n** Reading Pbias **\n");

		exit = ReadLandscape("qE_Pbias.txt", Pbias, div);
		if (exit)
			goto END_PROGRAM_1;
	}

	//*************************************************************************
	//********************* Metropolis Simulation *****************************
	//*************************************************************************

	if (abs(mode) <= 2)		// skip if abs(mode) > 2
	{
		printf("\n\n** Metropolis Monte Carlo **\n");

		seed = GetSeed();
		vslNewStream(&stream, rng, seed);

		for (tr = 0; tr <= ntr; tr++)
		{
			if (tr == 0)	// pick initial conditions if tr = 0
			{
				prob = 0.5;
				InitState(st, grd, prob, &idum, stream);

				printf("\nConditioning pulse...\n");

				exit = Ising_MA(st, P, Pbias, A_2, A_1, A0, A1, A2, B_2, B_1, B0, B1, B2, npts / 100, grd, div, &idum, stream);

				if (exit)
					goto END_PROGRAM_1;

				for (i = div.n1; i <= div.n2; i++)
					for (j = div.n1; j <= div.n2; j++)
					{
						P[i][j] = 0.0;
						A_2[i][j] = 0.0;
						A_1[i][j] = 0.0;
						A0[i][j] = 0.0;
						A1[i][j] = 0.0;
						A2[i][j] = 0.0;
						B_2[i][j] = 0.0;
						B_1[i][j] = 0.0;
						B0[i][j] = 0.0;
						B1[i][j] = 0.0;
						B2[i][j] = 0.0;
					}
			}	//equilibration pre-pulse

			printf("\nMA trace %d/%d\n", tr + 1, ntr + 1);

			// perform MC simulation
			exit = Ising_MA(st, P, Pbias, A_2, A_1, A0, A1, A2, B_2, B_1, B0, B1, B2, npts, grd, div, &idum, stream);
			if (exit)
				goto END_PROGRAM_1;
		} // tr

		  // compute rates
		for (i = div.n1; i <= div.n2; i++)
			for (j = div.n1; j <= div.n2; j++)
				if (P[i][j] != 0.0)
				{
					A_2[i][j] /= P[i][j];
					A_1[i][j] /= P[i][j];
					A0[i][j] /= P[i][j];
					A1[i][j] /= P[i][j];
					A2[i][j] /= P[i][j];
					B_2[i][j] /= P[i][j];
					B_1[i][j] /= P[i][j];
					B0[i][j] /= P[i][j];
					B1[i][j] /= P[i][j];
					B2[i][j] /= P[i][j];
				}

		// normalize P(na,nE) and W (Omega).

		Z = 0.0;
		for (i = div.n1; i <= div.n2; i++)
			for (j = div.n1; j <= div.n2; j++)
				Z += P[i][j];

		lnZ = log(Z);

		for (i = div.n1; i <= div.n2; i++)
			for (j = div.n1; j <= div.n2; j++)
				if (P[i][j])
					lng[i][j] = ndiv*log(2) + log(P[i][j]) - lnZ;

		lng[div.m1][div.n1] = 1e-10;
		lng[div.m2][div.n1] = 1e-10;

		SaveLandscape("qE_lng.txt", u, u, lng, div);
		SaveLandscape("qE_A_2.txt", u, u, A_2, div);
		SaveLandscape("qE_A_1.txt", u, u, A_1, div);
		SaveLandscape("qE_A0.txt", u, u, A0, div);
		SaveLandscape("qE_A1.txt", u, u, A1, div);
		SaveLandscape("qE_A2.txt", u, u, A2, div);
		SaveLandscape("qE_B_2.txt", u, u, B_2, div);
		SaveLandscape("qE_B_1.txt", u, u, B_1, div);
		SaveLandscape("qE_B0.txt", u, u, B0, div);
		SaveLandscape("qE_B1.txt", u, u, B1, div);
		SaveLandscape("qE_B2.txt", u, u, B2, div);

		if (mode < 0)
		{
			printf("\n\nCompleted Metropolis Monte Carlo. Press any key to exit.\n");
			goto END_PROGRAM_1;
		}
	}
	else			// read MA results from files, abort if any file does not exist
	{
		printf("\n\n** Reading rate matrices **\n");

		exit = ReadLandscape("qE_lng.txt", lng, div);
		exit += ReadLandscape("qE_A_2.txt", A_2, div);
		exit += ReadLandscape("qE_A_1.txt", A_1, div);
		exit += ReadLandscape("qE_A0.txt", A0, div);
		exit += ReadLandscape("qE_A1.txt", A1, div);
		exit += ReadLandscape("qE_A2.txt", A2, div);
		exit += ReadLandscape("qE_B_2.txt", B_2, div);
		exit += ReadLandscape("qE_B_1.txt", B_1, div);
		exit += ReadLandscape("qE_B0.txt", B0, div);
		exit += ReadLandscape("qE_B1.txt", B1, div);
		exit += ReadLandscape("qE_B2.txt", B2, div);
		if (exit)
			goto END_PROGRAM_1;
	}

	//*************************************************************************
	//********************* Compute Free energy landscape ********************
	//*************************************************************************

  	printf("\n\n** Free energy landscape **\n");

   // Check W against Ones matrix
	for (i = div.m1; i <= div.m2; i++)		//na
		for (j = div.n1; j <= div.m2; j++)		//nE
			if (Ones[i][j] ^ (fabs(lng[i][j]) > 0.0 ? 1 : 0))				// exclusive or - compares entries between Ones and W
			{
				printf("Ones and lng mismatch at na = %d, nE = %d). Press any key to end.\n", i, j);
				SaveLandscape("ERROR_lng.txt", u, u, lng, div);
				SaveIntLandscape("ERROR_Ones.txt", u, u, Ones, div);
				goto END_PROGRAM_1;
			}

    // free energy F(T,V)
    for (na = div.m1; na <= div.m2; na++)
        for (nE = div.n1; nE <= div.n2; nE++)
            if (Ones[na][nE] == 1)
                F[na][nE] = E[nE] - kT*lng[na][nE] - (q[na]-qmax/2)*V;

    // Truncate free energy and map (na,nE) -> state
    Fmin = 1e12;
    for (nE = div.n1; nE <= div.n2; nE++)
        if (Ones[ndiv/2][nE] == 1)
            if (F[ndiv/2][nE] < Fmin)
                Fmin = F[ndiv/2][nE];

    Fmax = trunc_height?Fmin+trunc_height:Fmax;       //if trunc_height is zero, set to Fmax

    nst = 0;
    m = 0;
    for (na = div.m1; na <= div.m2; na++)
    {
        ii = 0;
        for (nE = div.n1; nE <= div.n2; nE++)
            if (F[na][nE] < Fmax && Ones[na][nE] == 1)
            {
                nst++;
                ii++;
                state[na][nE] = nst;
            }
            else
                F[na][nE] = Fmax;
        m = (ii > m) ? ii : m;
    }
    m += 2;		// this is the half width of the band matrix

    SaveLandscape("qE_F.txt", u, u, F, div);
    SaveIntLandscape("qE_state.txt", u, u, state, div);

   // indices
    sts.m1 = 1;
    sts.m2 = nst;
    sts.n1 = 1;
    sts.n2 = nst;

    bnd.m1 = 1;
    bnd.m2 = nst;
    bnd.n1 = 1;
    bnd.n2 = 2 * m + 1;

    Na = MakeIntVector(sts);
    NE = MakeIntVector(sts);
    uu = MakeVector(sts);
    stq = MakeVector(sts);
    stE = MakeVector(sts);
	Q = MakeMatrix(bnd);
	Qsym = MakeMatrix(bnd);
	PqE = MakeVector(sts);

    ii = 0;
    for (na = div.m1; na <= div.m2; na++)
        for (nE = div.n1; nE <= div.n2; nE++)
            if (state[na][nE])
            {
                ii++;
                uu[ii] = ii;
                Na[ii] = na;
                NE[ii] = nE;
                stq[ii] = q[na];
                stE[ii] = E[nE];
            }

    Z = 0.0;							// partition function for truncated F
    for (ii = sts.m1; ii <= sts.m2; ii++)
    {
        na = Na[ii];
        nE = NE[ii];
        Z += (PqE[ii] = exp(-beta*F[na][nE]));
        Pq[na] += PqE[ii];
    }

    for (ii = sts.m1; ii <= sts.m2; ii++)
        PqE[ii] /= Z;

    for (na = div.m1; na <= div.m2; na++)
        Pq[na] /= Z;

    for (na=div.n1;na<=div.n2;na++)
		if(Pq[na] != 0.0)
			Wq[na]= (-kT*log(Pq[na]));
        else
            Wq[j]=0.0;

	for (na=div.n2;na>=div.n1;na--)
		Wq[na] -= Wq[div.n1];

    SaveVector("q_W.txt",q,Wq,div);
    SaveVector("q_P.txt",q,Pq,div);

    printf("\nFmax = %g. %d/%d states used. m = %d.\n", Fmax, nst, Ncells, m);

	//*************************************************************************
	//********************* Compute rate matrix *******************************
	//*************************************************************************

    printf("\n\n** Rate matrix **\n");

    for (ii = sts.m1; ii <= sts.m2; ii++)
    {
        na = Na[ii];
        nE = NE[ii];

        // na -> na+1; nE -> nE-2
        Calc_Rate_Pair (Q,Qsym,aq,bq,A_2,na,nE,B2,na+1,nE-2,PqE,state,ii,stq,stE,m);

        // na -> na+1; nE -> nE-1
        Calc_Rate_Pair (Q,Qsym,aq,bq,A_1,na,nE,B1,na+1,nE-1,PqE,state,ii,stq,stE,m);

        // na -> na+1; nE -> nE
        Calc_Rate_Pair (Q,Qsym,aq,bq,A0,na,nE,B0,na+1,nE,PqE,state,ii,stq,stE,m);

        // na -> na+1; nE -> nE+1
        Calc_Rate_Pair (Q,Qsym,aq,bq,A1,na,nE,B_1,na+1,nE+1,PqE,state,ii,stq,stE,m);

        // na -> na+1; nE -> nE+2
        Calc_Rate_Pair (Q,Qsym,aq,bq,A2,na,nE,B_2,na+1,nE+2,PqE,state,ii,stq,stE,m);
    }

    for (na = div.n1; na <= div.n2; na++)
        if (Pq[na] != 0.0)
        {
            aq[na] /= Pq[na];
            bq[na] /= Pq[na];
        }

    // calculate W(q) from aq and bq
	Wq[div.n1] = 0.0;
	for (na=div.n1+1;na<=div.n2;na++)
		if(aq[na-1] > 0 && bq[na] > 0)
			Wq[na] = Wq[na-1]-kT*log(aq[na-1]/bq[na]);

    SaveRateModel("q_IsingRate.txt",q,aq,bq,Wq,div);
    //SaveMatrix("Q.txt",uu,uu,Q,bnd);

 	//*************************************************************************
	//********************* Calculate MFPTs ***********************************
	//*************************************************************************

    printf("\n\n** MFPT **\n");

    //2D landscape

    // determine charge positions at min probabilities
    tmp = 0.0;
    st1 = sts.n1;
    for (ii = sts.n1; ii <= sts.n2/2; ii++)
        if (tmp < PqE[ii])
        {
            tmp = PqE[ii];
            st1 = ii;
        }

    tmp = 0.0;
    st2 = sts.n2;
    for (ii = sts.n2/2; ii <= sts.n2; ii++)
        if (tmp < PqE[ii])
        {
            tmp = PqE[ii];
            st2 = ii;
        }    for (na = div.m1; na <= div.m2; na++)
        q[na] = (na+1)*delq;


	st1q = Na[st1]*delq;
	st2q = Na[st2]*delq;

    tst.m1 = 1;
    tst.m2 = st2;
    tst.n1 = 1;
    tst.n2 = st2;

    fpt.m1 = 1;
    fpt.m2 = st2;
    fpt.n1 = bnd.n1;
    fpt.n2 = bnd.n2;

    indx = MakeIntVector(tst);
    neg1 = MakeVector(tst);
    Ql = MakeMatrix(bnd);
    Qc = MakeMatrix(bnd);

    for (ii = bnd.m1; ii <= bnd.m2; ii++)
        for (jj = bnd.n1; jj <= bnd.n2; jj++)
            Qc[ii][jj] = Q[ii][jj];

    for (ii = tst.n1; ii <= tst.n2; ii++)
        neg1[ii] = -1.0;

    // the following changes Q (!)
    bandec(Qc, m, m, Ql, indx, fpt);
    banbks(Qc, m, m, Ql, indx, neg1, fpt);

    if (neg1[st1] != 0.0)
        a2D = 1.0/neg1[st1];
    else a2D = 0.0;

    for (ii = sts.m1; ii <= sts.m2; ii++)
        uu[ii] = stq[ii];

    SaveVector("qE_P.txt",uu,PqE,sts);
    SaveVector("qE_mfpt.txt", uu, neg1, tst);

    for (ii = sts.m1; ii <= sts.m2; ii++)
        uu[ii] = ii;

    FreeIntVector(indx, tst);
    FreeVector(neg1, tst);
    FreeMatrix(Ql, bnd);
    FreeMatrix(Qc, bnd);

    //1D landscape

    tmp = 0.0;
    na1 = div.n1;
    for (na = div.n1; na <= div.n2/2; na++)
        if (tmp < Pq[na])
        {
            tmp = Pq[na];
            na1 = na;
        }

    tmp = 0.0;
    na2 = div.n2;
    for (na = div.n2/2; na <= div.n2; na++)
        if (tmp < Pq[na])
        {
            tmp = Pq[na];
            na2 = na;
        }

    if (na2 == div.n2)
    {
        printf("Error! na2 = ndiv.\n");
        goto END_PROGRAM_2;
   }

    K = 0.0;
    for (na = div.n1; na < na1; na++)
            K += Pq[na];

    tmp = 1.0;
    nab = na1;
    for (na = na1; na <= na2; na++)
        if (tmp > Pq[na])
        {
            tmp = Pq[na];
            nab = na;
            K += tmp;
        }

    na1q = na1*delq;
    nabq = nab*delq;
	na2q = na2*delq;
	K = (1.0-K)/K;

    tst.m1 = 1;
    tst.m2 = na2 + 1;
    tst.n1 = 1;
    tst.n2 = na2 + 1;

    fpt.m1 = 1;
    fpt.m2 = na2 + 1;
    fpt.n1 = 1;
    fpt.n2 = 3;

    tdi.m1 = 1;
    tdi.m2 = ndiv+1;
    tdi.n1 = 1;
    tdi.n2 = 3;

    indx = MakeIntVector(tst);
    neg1 = MakeVector(tst);
    Qc = MakeMatrix(tdi);
    Ql = MakeMatrix(fpt);

    //construct tridiagonal rate matrix
    for (na = div.m1; na <= div.m2; na++)
    {
        Qc[na+1][1] = bq[na];
        Qc[na+1][2] = -(aq[na] + bq[na]);
        Qc[na+1][3] = aq[na];
    }

    for (ii = tst.n1; ii <= tst.n2; ii++)
        neg1[ii] = -1.0;

    bandec(Qc, 1, 1, Ql, indx, fpt);
    banbks(Qc, 1, 1, Ql, indx, neg1, fpt);

    if (neg1[na1+1] != 0.0)
        a1D = 1.0/neg1[na1+1];
    else a1D = 0.0;

    for (na = div.m1; na <= div.m2; na++)
        q[na] = (na-1)*delq;

    SaveVector("q_mfpt.txt", q, neg1, tst);

    for (na = div.m1; na <= div.m2; na++)
        q[na] = na*delq;

    FreeIntVector(indx, tst);
    FreeVector(neg1, tst);
    FreeMatrix(Qc, tdi);
    FreeMatrix(Ql, fpt);

	//*************************************************************************
	//********************* Eigenvalues ************************************
	//*************************************************************************

	if (abs(mode) <= 3)		// skip if abs(mode) > 3
	{
		printf("\n\n** Eigenvalues **\n");

		// 2D eigenvalues
		ev = MakeVector(sts);

		EigenRate(ev, Qsym, bnd);
		EigSort_(ev, sts);

		for (ii = sts.n1; ii <= sts.n2; ii++)
			ev[ii] = -ev[ii];

		a_ev_2D = ev[sts.n1 + 1] * K / (1 + K);

		SaveVector("qE_ev.txt", uu, ev, sts);

		FreeVector(ev, sts);

		// 1D eigenvalues
		tdi.m1 = 0;
		tdi.m2 = 2;
		tdi.n1 = div.n1;
		tdi.n2 = div.n2;

		lEV = MakeMatrix(div);
		rEV = MakeMatrix(div);
		ev = MakeVector(div);
		Qc = MakeMatrix(tdi);

		//construct tridiagonal rate matrix
		for (na = tdi.n1; na < tdi.n2; na++)
			Qc[tdi.m2][na] = Qc[tdi.m1][na + 1] = sqrt(aq[na] * bq[na + 1]);

		//diagonals
		for (na = tdi.n1; na <= tdi.n2; na++)
			Qc[tdi.m1 + 1][na] = -(aq[na] + bq[na]);

		//SaveMatrix("Qsym.txt", u, u, Qc, tdi);

		EigenSystemXDE(lEV, rEV, ev, Qc, Pq, tdi);

		for (i = div.n1; i <= div.n2; i++)
			ev[i] = -ev[i];

		a_ev_1D = ev[div.n1 + 1] * K / (1 + K);

		SaveVector("q_ev.txt", u, ev, div);

		FreeMatrix(lEV, div);
		FreeMatrix(rEV, div);
		FreeVector(ev, div);
		FreeMatrix(Qc, tdi);

		if (mode < 0)
		{
			SaveResults("Eig_Results.txt", Ncells, nst, m, Fo, Hcheck, na1, na1q, nab, nabq, na2, na2q, st1, st1q, st2, st2q, 0, 0, 0, 0,
				0, 0, 0, 0, 0, 0, 0, 0, K, a2D, a1D, a_ev_1D, a_ev_2D);

			printf("\n\nCompleted Eigenvalues. Press any key to exit.\n");
			goto END_PROGRAM_2;
		}
	}
	else			// read eigenvalues from files, abort if any file does not exist
	{
		printf("\n\n** Reading eigenvalues **\n");

		ev = MakeVector(sts);
		exit = ReadVector("qE_ev.txt", ev, sts);
		a_ev_2D = ev[sts.n1 + 1] * K / (1 + K);
		FreeVector(ev, sts);

		ev = MakeVector(div);
		exit += ReadVector("q_ev.txt", ev, div);
		a_ev_1D = ev[div.n1 + 1] * K / (1 + K);
		FreeVector(ev, div);

		if (exit)
			goto END_PROGRAM_2;
	}

	//*************************************************************************
	//********************* 2D Monte Carlo Simulation *************************
	//*************************************************************************

	printf("\n\n** 2D Monte Carlo simulation **\n");

	acount2D = bcount2D = 0;
	amfpt2D = bmfpt2D = 0.0;

    //destroy contents of "times.txt" before first trace
    if((fp = fopen("qE_times.txt","w")) == NULL)
       goto END_PROGRAM_2;
    else
    {
        fprintf(fp,"%16s %16s %16s\n","(ms)","crossings","rate");
        fclose(fp);
    }

    for(tr=0;tr<=ntr;tr++)
	{
		if(tr==0)	// pick initial conditions and run short equilibration if tr = 0
		{
			ii = nst/2;

			printf("\nConditioning pulse...\n");

			Master_MC("qE_times.txt",&ii,Q,Na,npts/100,st1q,st2q,&acount2D,&amfpt2D,&bcount2D,&bmfpt2D,bnd, &idum, stream);
		}	//equilibration pre-pulse

		printf("\n2D MC trace %d/%d\n",tr+1,ntr+1);

    // perform MC simulation
        Master_MC("qE_times.txt",&ii,Q,Na,npts,st1q,st2q,&acount2D,&amfpt2D,&bcount2D,&bmfpt2D,bnd, &idum, stream);
	} // tr

    //*************************************************************************
	//********************* 1D Monte Carlo Simulation *************************
	//*************************************************************************

	printf("\n\n** 1D Monte Carlo simulation **\n");

	acount1D = bcount1D = 0;
	amfpt1D = bmfpt1D = 0.0;

    //destroy contents of "times1D.txt" before first trace
    if((fp = fopen("q_times.txt","w")) == NULL)
        goto END_PROGRAM_2;
    else
    {
        fprintf(fp,"%16s %16s %16s\n","(ms)","crossings","rate");
        fclose(fp);
    }

    tdi.m1 = 1;
    tdi.m2 = ndiv+1;
    tdi.n1 = 1;
    tdi.n2 = 3;

    tst.m1 = 1;
    tst.m2 = ndiv+1;
    tst.n1 = 1;
    tst.n2 = ndiv+1;

    iu = MakeIntVector(tst);
    Qc = MakeMatrix(tdi);

    //construct tridiagonal rate matrix
    for (na = div.m1; na <= div.m2; na++)
    {
        iu[na+1] = na;
        Qc[na+1][1] = bq[na];
        Qc[na+1][2] = -(aq[na] + bq[na]);
        Qc[na+1][3] = aq[na];
    }
    //SaveMatrix("Qtdi.txt",uu,uu,Qc,tdi);

    for(tr=0;tr<=ntr;tr++)
	{
		if(tr==0)	// pick initial conditions and run short equilibration if tr = 0
		{
			na = ndiv/2;

			printf("\nConditioning pulse...\n");

			Master_MC("q_times.txt",&na,Qc,iu,npts/100,na1q,na2q,&acount1D,&amfpt1D,&bcount1D,&bmfpt1D,tdi, &idum, stream);
 		}	//equilibration pre-pulse

		printf("\n1D MC trace %d/%d\n",tr+1,ntr+1);

    // perform MC simulation
        Master_MC("q_times.txt",&na,Qc,iu,npts,na1q,na2q,&acount1D,&amfpt1D,&bcount1D,&bmfpt1D,tdi, &idum, stream);
	} // tr

    FreeMatrix(Qc, tdi);
    FreeIntVector(iu,tst);

	//*************************************************************************
	//********************* kinetic Monte Carlo Simulation ********************
	//*************************************************************************

 	printf("\n\n** Kinetic Monte Carlo (native) **\n");

	acount = bcount = 0;
	amfpt = bmfpt = 0.0;

    //destroy contents of "times.txt" before first trace
    if((fp = fopen("times.txt","w")) == NULL)
        goto END_PROGRAM_2;
    else
    {
        fprintf(fp,"%16s %16s %16s\n","(ms)","crossings","rate");
        fclose(fp);
    }

	seed = GetSeed();
	vslNewStream(&stream, rng, seed);

    for(tr=0;tr<=ntr;tr++)
	{
		if(tr==0)	// pick initial conditions and run short equilibration if tr = 0
		{
			prob = 0.5;
			InitState (st, grd, prob, &idum, stream);

			printf("\n\nConditioning pulse...\n");

			Ising_MC(st,npts/100,tq,aq,bq,E,tqE,na1q,na2q,&acount,&amfpt,&bcount,&bmfpt,grd,div, &idum,stream);

			for (na=div.m1;na<=div.m2;na++)
			{
				tq[na] = 0.0;
				aq[na] = 0.0;
				bq[na] = 0.0;
				E[na] = 0.0;
			}

			for (na=div.m1;na<=div.m2;na++)
				for (nE=div.n1;nE<=div.n2;nE++)
					tqE[na][nE] = 0.0;		// comment out if starting pathway desired
		}	//equilibration pre-pulse

		printf("\nkMC trace %d/%d\n",tr+1,ntr+1);

    // perform MC simulation
		Ising_MC(st,npts,tq,aq,bq,E,tqE,na1q,na2q,&acount,&amfpt,&bcount,&bmfpt,grd,div, &idum,stream);
	} // tr

    tmp = 0.0;
    for (na=div.m1;na<=div.m2;na++)
        if(tq[na])
        {
            tmp += tq[na];
            aq[na] /= tq[na];
            bq[na] /= tq[na];
            E[na] /= tq[na];
        }

    SaveVector("E.txt",q,E,div);

    // calculate W(q) from aq and bq
	Wq[div.n1] = 0.0;
	for (na=div.m1+1;na<=div.m2;na++)
		if(aq[na-1] > 0 && bq[na] > 0)
			Wq[na] = Wq[na-1]-kT*log(aq[na-1]/bq[na]);

    SaveRateModel("IsingRate.txt",q,aq,bq,Wq,div);

    // calculate W(q) based on time-sampled probability distributions

	for (na=div.m1;na<=div.m2;na++)
		if(tq[na] != 0.0)
			Wq[na] = (-kT*log(tq[na]/tmp));
        else
            Wq[na] =0.0;

	for (na=div.m2;na>=div.m1;na--)
		Wq[na] -= Wq[div.m1];

    SaveVector("W.txt",q,Wq,div);

    // calculate P(q,E);
    for (na=div.m1;na<=div.m2;na++)
        for (nE=div.n1;nE<=div.n2;nE++)
			tqE[na][nE] /= tmp;

    SaveLandscape("P.txt",u,u,tqE,div);

	//*************************************************************************
	//********************* Save Results and End ******************************
	//*************************************************************************

	SaveResults("Results.txt", Ncells, nst, m, Fo, Hcheck, na1, na1q, nab, nabq, na2, na2q, st1, st1q, st2, st2q, acount, bcount, amfpt, bmfpt, 
		acount2D, bcount2D, amfpt2D, bmfpt2D, acount1D, bcount1D, amfpt1D, bmfpt1D, K, a2D, a1D, a_ev_1D, a_ev_2D);

	printf("\n\nIsing successfully completed. Press any key to exit\n");

END_PROGRAM_2:

    FreeIntVector(Na,sts);
    FreeIntVector(NE,sts);
    FreeVector(uu,sts);
    FreeVector(stq,sts);
    FreeVector(stE,sts);
    FreeVector(PqE,sts);
	FreeMatrix(Q,bnd);
	FreeMatrix(Qsym,bnd);

END_PROGRAM_1:

    FreeIntMatrix(st,grd);
    FreeIntMatrix(state,div);
    FreeIntMatrix(Ones,div);
    FreeVector(u,div);
    FreeVector(q,div);
    FreeVector(E,div);
    FreeVector(tq,div);
    FreeVector(aq,div);
    FreeVector(bq,div);
    FreeVector(Wq,div);
    FreeMatrix(tqE,div);
    FreeMatrix(lng_WL,div);
    FreeMatrix(Pbias,div);
 	FreeMatrix(P,div);
    FreeVector(Pq,div);
	FreeMatrix(lng,div);
 	FreeMatrix(A_2,div);
 	FreeMatrix(A_1,div);
 	FreeMatrix(A0,div);
 	FreeMatrix(A1,div);
 	FreeMatrix(A2,div);
 	FreeMatrix(B_2,div);
 	FreeMatrix(B_1,div);
 	FreeMatrix(B0,div);
 	FreeMatrix(B1,div);
 	FreeMatrix(B2,div);
	FreeMatrix(F,div);

	getchar();

	return(0);
}

void InitState (IMATRIX st, INDEX grd, REAL prob, long *idum, VSLStreamStatePtr stream)
{
	INTEGER i,j;
	REAL rn;

	for(i=grd.m1;i<=grd.m2;i++)
		for (j = grd.n1; j <= grd.n2; j++)
		{
			if (rng_mode == 3)
				rn = RandomNumber(idum);
			else
				vdRngUniform(METHOD, stream, 1, &rn, 0.0, 1.0);
			if (rn <= prob)			//start middle of (q,E) plot (random)
									//if((i+j)%2 == 0)						//start at (qmax/2,Emax) (checkerboard)
									//if(i<=(grid_length/2) == 0)			//start at (qmax/2,Emin) (split-screen)
				st[i][j] = 1;		// active state
			else
				st[i][j] = 0;		// resting state
		}
}

INTEGER ActiveCells (IMATRIX st, INDEX grd)
{
	INTEGER i,j,sum=0;

	for(i=grd.m1;i<=grd.m2;i++)
		for(j=grd.n1;j<=grd.n2;j++)
			sum += st[i][j];

	return sum;
}

INTEGER NumActive (INTEGER i, INTEGER j, IMATRIX st, INDEX grd)
{
	return st[i+1>grd.m2?grd.m1:i+1][j] + st[i-1<grd.m1?grd.m2:i-1][j] + st[i][j+1>grd.n2?grd.n1:j+1] + st[i][j-1<grd.n1?grd.n2:j-1];
}

INTEGER WangLandau(IMATRIX st, MATRIX log_g, VECTOR q, VECTOR E, REAL Fo, REAL F_final, INTEGER Hcheck, INTEGER Ncells, INDEX grd, INDEX div, long *idum, VSLStreamStatePtr stream)
{
        INTEGER exitflag = 0, na = 0,nE = 0,nna,delnE,flip,i,j,ii,jj,irn,isum,Nsweeps,SweepCount,flag=0;
		long count;
		REAL F,t;
        REAL rn;
		INTEGER Ea[5];
		IMATRIX H;
		MATRIX normlog_g;

		H = MakeIntMatrix(div);
		normlog_g	= MakeMatrix(div);

		Nsweeps = Hcheck*SQR(grid_length);

		for(i=0;i<=4;i++)		// calculate discrete interaction energies as a function of number of activated neighbors
			Ea[i] = 4-i;											// activated state

// determine system energy and displacement (order parameter) for starting configuration st

		for(i=grd.m1;i<=grd.m2;i++)
			for(j=grd.n1;j<=grd.n2;j++)
				if(st[i][j])			//active state
				{
					na += 1;							//displacement
					nE += Ea[NumActive(i,j,st,grd)];	//energy
				}

		nE >>= 1;				//divide by two since transitions occur in increments of 2*eps

		count = 0;
		printf("\nF = %g\n", Fo);

        for(F = Fo; F >= F_final;)
        {
			if (flag && count%(100*Nsweeps) == 0)
				printf("\nF = %g\n", F);
			SweepCount = 0;
			for(;;)				//trial loop
			{
				// choose cell randomly
				if (rng_mode == 3)
					rn = RandomNumber(idum);
				else
					vdRngUniform(METHOD, stream, 1, &rn, 0.0, 1.0);

				irn = (INTEGER) floor(rn*SQR(grid_length));		// random integer from 0 to nst-1
				i = 1 + irn/grid_length;						// row (1 to gridlength)
				j = 1 + irn%grid_length;						// column (1 to gridlength)

				// decide if transition occurs
				if (rng_mode == 3)
					rn = RandomNumber(idum);
				else
					vdRngUniform(METHOD, stream, 1, &rn, 0.0, 1.0);

				flip = (1-2*st[i][j]);					// flip direction: +1 for R to A; -1 for A to R
				nna = NumActive(i,j,st,grd);			// number of active nearest neighbors
				delnE = flip*(2-nna);					// discrete energy change for trial flip (-2 to 2) = delE/2*eps

				if (rn <= exp(log_g[na][nE]-log_g[na+flip][nE+delnE]))				// Wang-Landau criterion for transition (natural log)
				{
					nE += delnE;						// update energy
					na += flip;							// update position
					st[i][j] ^= 1;						// flip cell
				}

				H[na][nE] += 1;
				log_g[na][nE] += F;
				count++;
				SweepCount++;
				t = ((REAL) count)/((REAL)SQR(grid_length));

				if(flag && SweepCount == SQR(grid_length))
				{
					F = 1/t;
					break;
				}
				else if (SweepCount == Nsweeps)
				{
					SweepCount = 0;

					isum = 0;
					// check for completed H
					for (ii=div.n1;ii<=div.n2;ii++)
						for (jj=div.n1;jj<=div.n2;jj++)
							if(H[ii][jj] != 0)
								isum++;

					if (isum == Ncells)		//done with iteration
					{
						for (ii=div.n1;ii<=div.n2;ii++)
							for (jj=div.n1;jj<=div.n2;jj++)
								H[ii][jj] = 0;

						if (F <= 1/t)
						{
							F = 1/t;
							flag = 1;
							printf("\nSwitching to 1/t mode...\n");
						}
						else
						{
							F /= 2.0;
							printf("\nF = %g\n", F);
						}
						break;
					}
				}
			} // for(;;)
		} //F_final

		FreeIntMatrix(H,div);
		FreeMatrix(normlog_g,div);
		return exitflag;
}

INTEGER Ising_MA (IMATRIX st, MATRIX P, MATRIX Pbias, MATRIX A_2, MATRIX A_1, MATRIX A0, MATRIX A1, MATRIX A2, MATRIX B_2, MATRIX B_1, MATRIX B0, MATRIX B1, MATRIX B2,
					INTEGER npts, INDEX grd, INDEX div, long *idum, VSLStreamStatePtr stream)
{
        INTEGER exitflag = 0, na = 0,nE = 0,nna,delnE,flip,i,j,irn,NA;
		REAL a_2,a_1,a0,a1,a2,b_2,b_1,b0,b1,b2;
		REAL dt = 1/fs, simdt;
        REAL rn,time,endtime;
		INTEGER Ea[5];

		time = 0;				// starting point
		endtime = npts*dt;			// end point
		simdt = 1/fi;

		for(i=0;i<=4;i++)		// calculate discrete interaction energies and forward and backward rates as a function of number of activated neighbors
			Ea[i] = 4-i;											// activated state

// determine system energy and displacement (order parameter) for starting configuration st

		for(i=grd.m1;i<=grd.m2;i++)
			for(j=grd.n1;j<=grd.n2;j++)
				if(st[i][j])			//active state
				{
					na += 1;			//displacement
					nE += Ea[NumActive(i,j,st,grd)];
				}

		nE >>= 1;				//divide by two  to avoid counting twice

// begin simulation

        for(;;)
        {
			a_2 = a_1 = a0 = a1 = a2 = b_2 = b_1 = b0 = b1 = b2 = 0.0;

			for(i=grd.m1;i<=grd.m2;i++)
				for(j=grd.n1;j<=grd.n2;j++)
				{
					NA = NumActive(i,j,st,grd);
					if (st[i][j])								// activated state
					{
						if (NA == 0)
							b_2 += 1.0;
						else if (NA == 1)
							b_1 += 1.0;
						else if (NA == 2)
							b0 += 1.0;
						else if (NA == 3)
							b1 += 1.0;
						else if (NA == 4)
							b2 += 1.0;
					}
					else										// resting state
					{
						if (NA == 0)
							a2 += 1.0;
						else if (NA == 1)
							a1 += 1.0;
						else if (NA == 2)
							a0 += 1.0;
						else if (NA == 3)
							a_1 += 1.0;
						else if (NA == 4)
							a_2 += 1.0;
					}
				}

			time += simdt;
			P[na][nE] += Pbias[na][nE];
			A_2[na][nE] += a_2*Pbias[na][nE];
			B_2[na][nE] += b_2*Pbias[na][nE];
			A_1[na][nE] += a_1*Pbias[na][nE];
			B_1[na][nE] += b_1*Pbias[na][nE];
			A0[na][nE] += a0*Pbias[na][nE];
			B0[na][nE] += b0*Pbias[na][nE];
			A1[na][nE] += a1*Pbias[na][nE];
			B1[na][nE] += b1*Pbias[na][nE];
			A2[na][nE] += a2*Pbias[na][nE];
			B2[na][nE] += b2*Pbias[na][nE];

			if(time >= endtime)		// exit simulation
				goto EXIT_ROUTINE;

			// choose cell randomly
			if (rng_mode == 3)
				rn = RandomNumber(idum);
			else
				vdRngUniform(METHOD, stream, 1, &rn, 0.0, 1.0);

			irn = (INTEGER) floor(rn*SQR(grid_length));		// random integer from 0 to nst-1
			i = 1 + irn/grid_length;						// row (1 to grid_length)
			j = 1 + irn%grid_length;						// column (1 to grid_length)

			// decide if transition occurs
			if (rng_mode == 3)
				rn = RandomNumber(idum);
			else
				vdRngUniform(METHOD, stream, 1, &rn, 0.0, 1.0);

			flip = (1-2*st[i][j]);					// flip direction: +1 for R to A; -1 for A to R
			nna = NumActive(i,j,st,grd);			// number of active nearest neighbors
			delnE = flip*(2-nna);					// discrete energy change for trial flip (-2 to 2) = delE/2*eps

			if (Pbias[na+flip][nE+delnE] == 0.0)							//entering into new territory, create offset bias
				Pbias[na+flip][nE+delnE] = Pbias[na][nE];

			if (rn <= Pbias[na][nE]/Pbias[na+flip][nE+delnE])					// Metropolis criterion for transition
			{
				nE += delnE;						// update energy
				na += flip;							// update position
				st[i][j] ^= 1;						// flip cell
			}
		} // for (;;)

EXIT_ROUTINE:

		return exitflag;
}

INTEGER Ising_MC (IMATRIX st, INTEGER npts, VECTOR tq, VECTOR aq, VECTOR bq, VECTOR E, MATRIX tqE, REAL q1, REAL q2,
                  INTEGER *acount, REAL *amfpt, INTEGER *bcount, REAL *bmfpt, INDEX grd, INDEX div, long *idum, VSLStreamStatePtr stream)
{
        INTEGER i,j=0,tmpi,tmpj,na,nE,mstate;
		INTEGER exitflag = 0,stateflag = 0, ncrossings = 0,flip,nna;
		REAL kT, beta, delq, bo, dt, qb = 0.5*qmax, simq, simtime = 0.0,tau;
        REAL sum,tmp,ratesum,Energy,rn,tij,time,endtime,atmp,btmp;
		MATRIX Rate;
		REAL a[5],b[5];
		INTEGER Er[5],Ea[5];
		FILE *fp;

		Rate = MakeMatrix(grd);

		dt = 1/fs;
		time = 0;				// starting point
		endtime = npts*dt;			// end point

		bo = ao/Ko;
		delq = qmax/div.n2;
        kT = 0.086174*(TdegC+273.15);
        beta = (kT == 0.0)?0.0:1.0/kT;

		for(i=0;i<=4;i++)		// calculate discrete interaction energies and forward and backward rates as a function of number of activated neighbors
		{
			Er[i] = i;												// resting state
			Ea[i] = 4-i;											// activated state
			a[i] = ao*exp(-beta*x*((Ea[i]-Er[i])*eps-delq*V));			// forward rate constant
			b[i] = bo*exp(-beta*(x-1.0)*((Ea[i]-Er[i])*eps-delq*V));		// backward rate constant
		}

		// determine rate constants for starting configuration st
		for(i=grd.m1;i<=grd.m2;i++)
			for(j=grd.n1;j<=grd.n2;j++)
				Rate[i][j] = st[i][j]?b[NumActive(i,j,st,grd)]:a[NumActive(i,j,st,grd)];

		// determine discrete system energy and displacement (order parameter) for starting configuration st
		na = nE = 0;
		for(i=grd.m1;i<=grd.m2;i++)
			for(j=grd.n1;j<=grd.n2;j++)
				if(st[i][j])			//active state
				{
					na += 1;			                //displacement
					nE += Ea[NumActive(i,j,st,grd)];    //energy
				}

		nE >>= 1;				//divide by two since transitions occur in increments of 2*eps

		simq = na*delq;
		mstate = simq<=qb?-1:1;

// begin simulation

        for(;;)
        {
            ratesum = 0.0;
			atmp = 0.0;
			btmp = 0.0;

			for(i=grd.m1;i<=grd.m2;i++)
				for(j=grd.n1;j<=grd.n2;j++)
				{
					tmp = Rate[i][j];
					if (st[i][j])							// activated state
						btmp += tmp;
					else									// resting state
						atmp += tmp;
					ratesum += tmp;
				}
			if (rng_mode == 3)
				rn = RandomNumber(idum);
			else
				vdRngUniform(METHOD, stream, 1, &rn, 0.0, 1.0);
			tij = -log(rn)/ratesum;

            sum = 0.0;
			if (rng_mode == 3)
				rn = RandomNumber(idum);
			else
				vdRngUniform(METHOD, stream, 1, &rn, 0.0, 1.0);

			for(i=grd.m1;i<=grd.m2;i++)
				for(j=grd.n1;j<=grd.n2;j++)
				{
					sum += Rate[i][j]/ratesum;
					if (rn <= sum)
						goto BREAK_LOOP;	// transition cell (i,j) determined. Caution: do NOT use i or j as dummy variable until cell update completed.
				}
			BREAK_LOOP:

			Energy = nE*2*eps - na*delq*V;
			time += tij;
			tq[na] += tij;
			tqE[na][nE] += tij;
			E[na] += tij*Energy;
			aq[na] += tij*atmp;
			bq[na] += tij*btmp;

			flip = (1-2*st[i][j]);					// flip direction: +1 for R to A; -1 for A to R
			nna = NumActive(i,j,st,grd);			// number of active nearest neighbors

			na += flip;						// update position
			nE += flip*(2-nna);				// update energy
			simq += flip*delq;

			if (mstate==-1 && simq>qb)
			{
				ncrossings++;
				mstate = 1;
			}
			else if (mstate==1 && simq<=qb)
			{
				ncrossings++;
				mstate = -1;
			}

			if(simq<q1 && stateflag>=0)    //crossing q1 for first time since: 1) last q2 crossing, or 2) start of simulation
			{
				if(stateflag!=0)                // only if q2 was previously crossed
				{
				    tau = time-simtime;
				    (*bcount)++;
				    *bmfpt = *bmfpt + (tau-*bmfpt)/(*bcount);

					if((fp = fopen("times.txt","a")) == NULL)
                    {
                        exitflag = 1;
 						goto EXIT_ROUTINE;
                    }
					fprintf(fp,"%16g %16d %16s\n",tau,ncrossings,"b");
					fclose(fp);
				}
				ncrossings = 0;
				simtime = time;
				stateflag = -1;
			}
			else if(simq>q2 && stateflag<=0)   //crossing q2 for first time since: 1) last q1 crossing, or 2) start of simulation
			{
				if(stateflag!=0)                    // only if q1 was previously crossed
				{
				    tau = time-simtime;
				    (*acount)++;
				    *amfpt = *amfpt + (tau-*amfpt)/(*acount);

					if((fp = fopen("times.txt","a")) == NULL)
                    {
                        exitflag = 1;
 						goto EXIT_ROUTINE;
                    }
					fprintf(fp,"%16g %16d %16s\n",tau,ncrossings,"a");
					fclose(fp);
				}
				ncrossings = 0;
				simtime = time;
				stateflag = 1;
			}

			//flip cell

			st[i][j] ^= 1;

			if(time >= endtime)		// exit simulation
				goto EXIT_ROUTINE;

            // update rate constants of flipped cell and its immediate neighbors

            // target cell:
			Rate[i][j] = st[i][j]?b[NumActive(i,j,st,grd)]:a[NumActive(i,j,st,grd)];

            // lower neighbor
			tmpi = (i+1>grd.m2)?grd.m1:i+1;
			Rate[tmpi][j] = st[tmpi][j]?b[NumActive(tmpi,j,st,grd)]:a[NumActive(tmpi,j,st,grd)];

            // upper neighbor
			tmpi = (i-1<grd.m1)?grd.m2:i-1;
			Rate[tmpi][j] = st[tmpi][j]?b[NumActive(tmpi,j,st,grd)]:a[NumActive(tmpi,j,st,grd)];

            // right neighbor
			tmpj = (j+1>grd.n2)?grd.n1:j+1;
			Rate[i][tmpj] = st[i][tmpj]?b[NumActive(i,tmpj,st,grd)]:a[NumActive(i,tmpj,st,grd)];

            // left neighbor
			tmpj = (j-1<grd.n1)?grd.n2:j-1;
			Rate[i][tmpj] = st[i][tmpj]?b[NumActive(i,tmpj,st,grd)]:a[NumActive(i,tmpj,st,grd)];
		} // for (;;)


EXIT_ROUTINE:
		FreeMatrix(Rate,grd);

		return exitflag;
}

INTEGER Master_MC (CHARACTER* filename, INTEGER *st, MATRIX Q, IVECTOR na, INTEGER npts, REAL q1, REAL q2,
                  INTEGER *acount, REAL *amfpt, INTEGER *bcount, REAL *bmfpt, INDEX bnd, long *idum, VSLStreamStatePtr stream)
{
        INTEGER i, m, mstate,exitflag = 0,stateflag = 0, ncrossings = 0;
		REAL delq, dt, qb = 0.5*qmax, simq, simtime,tau,sum;
        REAL ratesum,rn,tij,time,endtime;
		FILE *fp;

		dt = 1/fs;
		time = simtime = 0.0;		// starting point
		endtime = npts*dt;			// end point

        m = (bnd.n2-1)/2;
		delq = qmax/SQR(grid_length);
		simq = na[*st]*delq;
		mstate = simq<=qb?-1:1;

// begin simulation

        for(;;)
        {
            ratesum = -Q[*st][m+1];
			if (rng_mode == 3)
				rn = RandomNumber(idum);
			else
				vdRngUniform(METHOD, stream, 1, &rn, 0.0, 1.0);
			tij = -log(rn)/ratesum;

			if (rng_mode == 3)
				rn = RandomNumber(idum);
			else
				vdRngUniform(METHOD, stream, 1, &rn, 0.0, 1.0);

            sum = 0.0;
            for (i=1; i<=2*m+1; i++)
            {
                if (i != m+1)
                {
                    sum += Q[*st][i];
                    if (rn <= sum/ratesum)
                        break;
                }
            }

			time += tij;
			*st += i-m-1;

			simq = na[*st]*delq;

			if (mstate==-1 && simq>qb)
			{
				ncrossings++;
				mstate = 1;
			}
			else if (mstate==1 && simq<=qb)
			{
				ncrossings++;
				mstate = -1;
			}

			if(simq<q1 && stateflag>=0)    //crossing q1 for first time since: 1) last q2 crossing, or 2) start of simulation
			{
				if(stateflag!=0)                // only if q2 was previously crossed
				{
				    tau = time-simtime;
				    (*bcount)++;
				    *bmfpt = *bmfpt + (tau-*bmfpt)/(*bcount);

					if((fp = fopen(filename,"a")) == NULL)
                    {
                        exitflag = 1;
 						goto EXIT_ROUTINE;
                    }
					fprintf(fp,"%16g %16d %16s\n",tau,ncrossings,"b");
					fclose(fp);
				}
				ncrossings = 0;
				simtime = time;
				stateflag = -1;
			}
			else if(simq>q2 && stateflag<=0)   //crossing q2 for first time since: 1) last q1 crossing, or 2) start of simulation
			{
				if(stateflag!=0)                    // only if q1 was previously crossed
				{
				    tau = time-simtime;
				    (*acount)++;
				    *amfpt = *amfpt + (tau-*amfpt)/(*acount);

					if((fp = fopen(filename,"a")) == NULL)
                    {
                        exitflag = 1;
 						goto EXIT_ROUTINE;
                    }
					fprintf(fp,"%16g %16d %16s\n",tau,ncrossings,"a");
					fclose(fp);
				}
				ncrossings = 0;
				simtime = time;
				stateflag = 1;
			}

			if(time >= endtime)		// exit simulation
				goto EXIT_ROUTINE;
		} // for (;;)

EXIT_ROUTINE:

		return exitflag;
}

void SavePar(void)
{
    FILE *fp;
	fp = fopen("par.mdl","w");
	fprintf(fp,"Parameter file for electrical Ising model.\n\n");
	fprintf(fp,"Environmental variables:\n\n");
	fprintf(fp,"%-15s%-10d%-10s%-10s%s\n","grid_length",grid_length,"-","cells","dimension of one side of simulation grid");
	fprintf(fp,"%-15s%-10g%-10s%-10s%s\n","qmax",qmax,"-","eu","total charge displacement");
	fprintf(fp,"%-15s%-10g%-10g%-10s%s\n","V",V,dV,"mV","applied voltage");
	fprintf(fp,"%-15s%-10g%-10g%-10s%s\n","TdegC",TdegC,dTdegC,"degC","temperature");
	fprintf(fp,"\nUnit cell kinetics:\n\n");
	fprintf(fp,"%-15s%-10g%-10g%-10s%s\n","ao",ao,logra,"kHz","forward rate constant");
	fprintf(fp,"%-15s%-10g%-10g%-10s%s\n","Ko",Ko,logrK,"","equilibrium constant");
	fprintf(fp,"%-15s%-10g%-10g%-10s%s\n","x",x,dx,"","LFER constant (0 to 1)");
	fprintf(fp,"%-15s%-10g%-10g%-10s%s\n","eps",eps,deps,"meV","interaction energy");
	fprintf(fp,"\nSimulation parameters:\n\n");
	fprintf(fp,"%-15s%-10g%-10s%s\n","fc",fc,"kHz","filter cut-off frequency");
	fprintf(fp,"%-15s%-10g%-10s%s\n","fs",fs,"kHz","sampling frequency");
	fprintf(fp,"%-15s%-10g%-10s%s\n","fi",fi,"kHz","Langevin sim frequency");
	fprintf(fp,"%-15s%-10d%-10s%s\n","filter_div",filter_div,"","number of impulse response phases");
	fprintf(fp,"%-15s%-10d%-10s%s\n","npts",npts,"","number of data points (0 to npts)");
	fprintf(fp,"%-15s%-10d%-10s%s\n","ntr",ntr,"","number of added traces (may be 0)");
	fprintf(fp,"\nBiased sampling:\n\n");
	fprintf(fp,"%-15s%-10g%-10g%-10s\n","window",w1,w2,"Enter left and right borders (0 to 1)");
	fprintf(fp,"%-15s%-10s%-4s%s\n","input_file",BiasFile,"","If file not found, sampling is unbiased");
	fprintf(fp,"\nPlot dimensions:\n\n");
	fprintf(fp,"%-15s%-10d%-10s%s\n","plot_xo",plot_xo,"pixels","x-value of plot origin");
	fprintf(fp,"%-15s%-10d%-10s%s\n","plot_yo",plot_yo,"pixels","y-value of plot origin");
	fprintf(fp,"%-15s%-10d%-10s%s\n","cell_length",cell_length,"pixels","length of one cell");
	fprintf(fp,"%-15s%-10d%-10s%s\n","grid_margin",grid_margin,"pixels","margin around grid");
	fprintf(fp,"\n%-15s%-10d%-10s%s\n","sim_Mode",mode,"","0 = all,(-)1-> W-L,(-)2-> mMC,(-)3-> EV,(-)4-> kMC");
	fprintf(fp,"%-15s%-10g%-10s%s\n","F_final",F_final,"","Wang-Landau end point");
	fprintf(fp,"%-15s%-10g%-10s%s\n","trunc_height",trunc_height,"meV","truncate above barrier peak");
	fprintf(fp,"%-15s%-10g%-10s%s\n","Fmax",Fmax,"meV","Free energy cut-off if trunc_height = 0");
	fprintf(fp, "%-15s%-10d%-10s%s\n", "RNG_Mode", rng_mode,"", "0-MT, 1-MC, 2-MRG, 3-ran2");
	fclose(fp);
}

INTEGER ReadPar(void)
{
	CHARACTER text[120];
	FILE *fp;
	if ( (fp = fopen("par.mdl","r")) == NULL )
		return 1;

    fgets(text,100,fp);		// "Parameter file for electrical Ising model.\n\n"
    fgets(text,100,fp);
    fgets(text,100,fp);		// "Environmental variables:\n\n"
    fgets(text,100,fp);
    fscanf(fp, "%s %d",text,&grid_length),fgets(text,100,fp);
    fscanf(fp, "%s %lf",text,&qmax),fgets(text,100,fp);
    fscanf(fp, "%s %lf %lf",text,&V,&dV),fgets(text,100,fp);
    fscanf(fp, "%s %lf %lf",text,&TdegC,&dTdegC),fgets(text,100,fp);
    fgets(text,100,fp);
    fgets(text,100,fp);
    fgets(text,100,fp);
    fscanf(fp, "%s %lf %lf",text,&ao,&logra),fgets(text,100,fp);
    fscanf(fp, "%s %lf %lf",text,&Ko,&logrK),fgets(text,100,fp);
    fscanf(fp, "%s %lf %lf",text,&x,&dx),fgets(text,100,fp);
    fscanf(fp, "%s %lf %lf",text,&eps,&deps),fgets(text,100,fp);
    fgets(text,100,fp);
    fgets(text,100,fp);
    fgets(text,100,fp);
    fscanf(fp, "%s %lf",text,&fc),fgets(text,100,fp);
    fscanf(fp, "%s %lf",text,&fs),fgets(text,100,fp);
    fscanf(fp, "%s %lf",text,&fi),fgets(text,100,fp);
    fscanf(fp, "%s %d",text,&filter_div),fgets(text,100,fp);
    fscanf(fp, "%s %d",text,&npts),fgets(text,100,fp);
    fscanf(fp, "%s %d",text,&ntr),fgets(text,100,fp);
    fgets(text,100,fp);
    fgets(text,100,fp);
    fgets(text,100,fp);
    fscanf(fp, "%s %lf %lf",text,&w1,&w2),fgets(text,100,fp);
    fscanf(fp, "%s %s",text,BiasFile),fgets(text,100,fp);
    fgets(text,100,fp);
    fgets(text,100,fp);
    fgets(text,100,fp);
    fscanf(fp, "%s %d",text,&plot_xo),fgets(text,100,fp);
    fscanf(fp, "%s %d",text,&plot_yo),fgets(text,100,fp);
    fscanf(fp, "%s %d",text,&cell_length),fgets(text,100,fp);
    fscanf(fp, "%s %d",text,&grid_margin),fgets(text,100,fp);
    fgets(text,100,fp);
    fscanf(fp, "%s %d",text,&mode),fgets(text,100,fp);
    fscanf(fp, "%s %lf",text,&F_final),fgets(text,100,fp);
   fscanf(fp, "%s %lf",text,&trunc_height),fgets(text,100,fp);
   fscanf(fp, "%s %lf",text,&Fmax),fgets(text,100,fp);
   fscanf(fp, "%s %d", text, &rng_mode), fgets(text, 100, fp);
   fclose(fp);

	return 0;
}

void SaveRateModel(CHARACTER* filename, VECTOR q, VECTOR aq, VECTOR bq, VECTOR W, INDEX sts)
{
	INTEGER i;
    FILE *fp;

	// save file
	fp = fopen(filename,"w");
    fprintf(fp,"%s\n", filename);
	fprintf(fp,"%-15s%-15s%-15s%-15s\n","q","alpha","beta","W");
	for(i=sts.n1;i<=sts.n2;i++)
		fprintf(fp,"%-15g%-15g%-15g%-15g\n",q[i],aq[i],bq[i],W[i]);
	fclose(fp);
}

void SaveIntLandscape(CHARACTER* filename, VECTOR q, VECTOR E, IMATRIX M, INDEX indx)
{
    FILE *fp;
    INTEGER i,j;

    if( (fp = fopen( filename, "w+" )) == NULL )
    {
            printf( "Error opening file\n" );
    }

    fprintf(fp,"%s\n", filename);
    fprintf(fp,"%-16s %dx%d\n", "Grid", grid_length, grid_length);
    fprintf(fp,"%-16s %g %-16s\n", "T", TdegC, "degC");
    fprintf(fp,"%-16s %g %-16s\n", "V", V, "mV");
    fprintf(fp,"%-16s %g %-16s\n", "eps", eps, "meV");
    fprintf(fp,"%-16s %g %-16s\n", "ao", ao, "kHz");

    fprintf(fp,"%-16s ", "E\\q");
    for (j=indx.n1;j<=indx.n2;j++)
        fprintf(fp, "%-16.8g ", q[j]);
    fprintf(fp,"\n");

    for (i=indx.m1;i<=indx.m2;i++)
    {
        fprintf(fp, "%-16.8g ", E[i]);
        for (j=indx.n1;j<=indx.n2;j++)
                fprintf(fp, "%-16d ", M[j][i]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void SaveMatrix(CHARACTER* filename, VECTOR u, VECTOR v, MATRIX M, INDEX indx)
{
    FILE *fp;
    INTEGER i,j;

    if( (fp = fopen( filename, "w+" )) == NULL )
    {
            printf( "Error opening file\n" );
    }

    fprintf(fp,"%s\n", filename);
    fprintf(fp,"%-16s %dx%d\n", "Grid", grid_length, grid_length);
    fprintf(fp,"%-16s %g %-16s\n", "T", TdegC, "degC");
    fprintf(fp,"%-16s %g %-16s\n", "V", V, "mV");
    fprintf(fp,"%-16s %g %-16s\n", "eps", eps, "meV");
    fprintf(fp,"%-16s %g %-16s\n", "ao", ao, "kHz");

    fprintf(fp,"%-16s ", "i\\j");
    for (j=indx.n1;j<=indx.n2;j++)
        fprintf(fp, "%-16.8g ", v[j]);
    fprintf(fp,"\n");

    for (i=indx.m1;i<=indx.m2;i++)
    {
        fprintf(fp, "%-16.8g ", u[i]);
        for (j=indx.n1;j<=indx.n2;j++)
                fprintf(fp, "%-16.8g ", M[i][j]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void SaveLandscape(CHARACTER* filename, VECTOR q, VECTOR E, MATRIX M, INDEX indx)
{
    FILE *fp;
    INTEGER i,j;

    if( (fp = fopen( filename, "w+" )) == NULL )
    {
            printf( "Error opening file\n" );
    }

	// header
	fprintf(fp, "%s %dx%d\n", filename, grid_length, grid_length);
    fprintf(fp,"%-16s %g\n", "T", TdegC);
    fprintf(fp,"%-16s %g\n", "V", V);
    fprintf(fp,"%-16s %g\n", "eps", eps);
    fprintf(fp,"%-16s %g\n", "ao", ao);
    fprintf(fp,"%-16s %g\n", "Fmax", Fmax);

	//first line
    fprintf(fp,"%-16s ", "E\\q");
    for (j=indx.n1;j<=indx.n2;j++)
        fprintf(fp, "%-16.8g ", q[j]);
    fprintf(fp,"\n");

	// remaining lines, matrix in reverse order
	for (i=indx.m1;i<=indx.m2;i++)
    {
        fprintf(fp, "%-16.8g ", E[i]);
        for (j=indx.n1;j<=indx.n2;j++)
                fprintf(fp, "%-16.8g ", M[j][i]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

INTEGER ReadLandscape(CHARACTER* filename, MATRIX M, INDEX indx)
{
	CHARACTER text[80];
	INTEGER i, j;
	REAL tmp;
	FILE *fp;
	if ((fp = fopen(filename, "r")) == NULL)
		return 1;
	
	// header
	fgets(text, 100, fp);
	fgets(text, 100, fp);
	fgets(text, 100, fp);
	fgets(text, 100, fp);
	fgets(text, 100, fp);
	fgets(text, 100, fp);

	// first line
	fscanf(fp, "%s", text);
	for (j = indx.n1; j <= indx.n2; j++)
		fscanf(fp, "%lf", &tmp);
	fgets(text, 100, fp);

	// remaining lines, matrix in reverse order
	for (i = indx.n1; i <= indx.n2; i++)
	{
		fscanf(fp, "%lf", &tmp);
		for (j = indx.n1; j <= indx.n2; j++)
			fscanf(fp, "%lf", &M[j][i]);
		fgets(text, 100, fp);
	}

	fclose(fp);
	return 0;
}

void SaveVector(CHARACTER* filename, VECTOR q, VECTOR vv, INDEX indx)
{
	FILE *fp;
	INTEGER i;

	if ((fp = fopen(filename, "w+")) == NULL)
	{
		printf("Error opening file\n");
	}

	fprintf(fp, "%s %dx%d\n", filename, grid_length, grid_length);
    fprintf(fp,"%-16s %g\n", "T", TdegC);
    fprintf(fp,"%-16s %g\n", "V", V);
    fprintf(fp,"%-16s %g\n", "eps", eps);
    fprintf(fp,"%-16s %g\n", "ao", ao);
    fprintf(fp,"%-16s %g\n", "Fmax", Fmax);

	for (i = indx.n1; i <= indx.n2; i++)
		fprintf(fp, "%-16.8g %-16.8g\n", q[i], vv[i]);

	fclose(fp);
}

INTEGER ReadVector(CHARACTER* filename, VECTOR vv, INDEX indx)
{
	CHARACTER text[80];
	INTEGER i;
	REAL tmp;
	FILE *fp;
	if ((fp = fopen(filename, "r")) == NULL)
		return 1;

	// header
	fgets(text, 100, fp);
	fgets(text, 100, fp);
	fgets(text, 100, fp);
	fgets(text, 100, fp);
	fgets(text, 100, fp);
	fgets(text, 100, fp);


	// data
	for (i = indx.n1; i <= indx.n2; i++)
	{
		fscanf(fp, "%lf %lf", &tmp, &vv[i]);
		fgets(text, 100, fp);
	}

	fclose(fp);
	return 0;
}

void SaveResults(CHARACTER* filename, INTEGER Ncells, INTEGER nst, INTEGER m, REAL Fo, INTEGER Hcheck, INTEGER na1, REAL na1q, INTEGER nab, REAL nabq, INTEGER na2, 
	REAL na2q, INTEGER st1, REAL st1q, INTEGER st2, REAL st2q, INTEGER acount, INTEGER bcount, REAL amfpt, REAL bmfpt, INTEGER acount2D, INTEGER bcount2D, 
	REAL amfpt2D, REAL bmfpt2D, INTEGER acount1D, INTEGER bcount1D, REAL amfpt1D, REAL bmfpt1D, REAL K, REAL a2D, REAL a1D, REAL a_ev_1D, REAL a_ev_2D)
{
	FILE *fp;
	INTEGER ndiv = SQR(grid_length);
	REAL delq = qmax / ndiv;
	REAL T = TdegC + 273.15;
	REAL kT = 0.086174*T;
	REAL beta = (kT == 0.0) ? 0.0 : 1.0 / kT;

	if ((fp = fopen(filename, "w+")) == NULL)
		printf("Failed to write to Results.txt.\n");
	else
	{
		fprintf(fp, "%-16s %-16d\n", "RNG", rng_mode);
		fprintf(fp, "%-16s %-16d\n", "L", grid_length);
		fprintf(fp, "%-16s %-16d\n", "N", ndiv);
		fprintf(fp, "%-16s %-16g %-16s\n", "TdegC", TdegC, "degC");
		fprintf(fp, "%-16s %-16g %-16s\n", "T", T, "Kelvin");
		fprintf(fp, "%-16s %-16g %-16s\n", "kT", kT, "meV");
		fprintf(fp, "%-16s %-16g %-16s\n", "beta", beta, "1/meV");
		fprintf(fp, "%-16s %-16g %-16s\n", "V", V, "mV");
		fprintf(fp, "%-16s %-16g %-16s\n", "qmax", qmax, "eu");
		fprintf(fp, "%-16s %-16g %-16s\n", "delq", qmax / ndiv, "eu");
		fprintf(fp, "%-16s %-16g %-16s\n", "eps", eps, "meV");
		fprintf(fp, "%-16s %-16g %-16s\n", "ao", ao, "kHz");
		fprintf(fp, "%-16s %-16g %-16s\n", "Fmax", Fmax, "meV");
		fprintf(fp, "%-16s %-16d\n", "Ncells", Ncells);
		fprintf(fp, "%-16s %-16d\n", "nst", nst);
		fprintf(fp, "%-16s %-16d\n", "m", m);
		fprintf(fp, "%-16s %-16g\n", "Fo", Fo);
		fprintf(fp, "%-16s %-16g\n", "F_final", F_final);
		fprintf(fp, "%-16s %-16d\n", "Hcheck", Hcheck);
		fprintf(fp, "%-16s %-16d\n", "na1", na1);
		fprintf(fp, "%-16s %-16g %-16s\n", "q1", na1q, "eu");
		fprintf(fp, "%-16s %-16d\n", "nab", nab);
		fprintf(fp, "%-16s %-16g %-16s\n", "qb", nabq, "eu");
		fprintf(fp, "%-16s %-16d\n", "na2", na2);
		fprintf(fp, "%-16s %-16g %-16s\n", "q2", na2q, "eu");
		fprintf(fp, "%-16s %-16d\n", "st1", st1);
		fprintf(fp, "%-16s %-16g %-16s\n", "q_1", st1q, "eu");
		fprintf(fp, "%-16s %-16d\n", "st2", st2);
		fprintf(fp, "%-16s %-16g %-16s\n", "q_2", st2q, "eu");
		fprintf(fp, "%-16s %-16d\n", "acount", acount);
		fprintf(fp, "%-16s %-16d\n", "bcount", bcount);
		fprintf(fp, "%-16s %-16g %-16s\n", "aMC", 1.0 / amfpt, "kHz");
		fprintf(fp, "%-16s %-16g %-16s\n", "bMC", 1.0 / bmfpt, "kHz");
		fprintf(fp, "%-16s %-16d\n", "acount2D", acount2D);
		fprintf(fp, "%-16s %-16d\n", "bcount2D", bcount2D);
		fprintf(fp, "%-16s %-16g %-16s\n", "aMC2D", 1.0 / amfpt2D, "kHz");
		fprintf(fp, "%-16s %-16g %-16s\n", "bMC2D", 1.0 / bmfpt2D, "kHz");
		fprintf(fp, "%-16s %-16d\n", "acount1D", acount1D);
		fprintf(fp, "%-16s %-16d\n", "bcount1D", bcount1D);
		fprintf(fp, "%-16s %-16g %-16s\n", "aMC1D", 1.0 / amfpt1D, "kHz");
		fprintf(fp, "%-16s %-16g %-16s\n", "bMC1D", 1.0 / bmfpt1D, "kHz");
		fprintf(fp, "%-16s %-16g\n", "K", K);
		fprintf(fp, "%-16s %-16g %-16s\n", "a_mfpt_2D", a2D, "kHz");
		fprintf(fp, "%-16s %-16g %-16s\n", "b_mfpt_2D", a2D / K, "kHz");
		fprintf(fp, "%-16s %-16g %-16s\n", "a_mfpt_1D", a1D, "kHz");
		fprintf(fp, "%-16s %-16g %-16s\n", "b_mfpt_1D", a1D / K, "kHz");
		fprintf(fp, "%-16s %-16g %-16s\n", "ev_1D", a_ev_1D*(1 + K) / K, "kHz");
		fprintf(fp, "%-16s %-16g %-16s\n", "a_ev_1D", a_ev_1D, "kHz");
		fprintf(fp, "%-16s %-16g %-16s\n", "b_ev_1D", a_ev_1D / K, "kHz");
		fprintf(fp, "%-16s %-16g %-16s\n", "ev_2D", a_ev_2D*(1 + K) / K, "kHz");
		fprintf(fp, "%-16s %-16g %-16s\n", "a_ev_2D", a_ev_2D, "kHz");
		fprintf(fp, "%-16s %-16g %-16s\n", "b_ev_2D", a_ev_2D / K, "kHz");
		fclose(fp);
	}
}

void bandec(MATRIX a, INTEGER m1, INTEGER m2, MATRIX al, IVECTOR indx, INDEX sts)
{
	INTEGER i, j, k, l, mm, n = sts.m2;
	REAL dum;

	mm = m1 + m2 + 1;
	l = m1;
	for (i = 1; i <= m1; i++) {
		for (j = m1 + 2 - i; j <= mm; j++) a[i][j - l] = a[i][j];
		l--;
		for (j = mm - l; j <= mm; j++) a[i][j] = 0.0;
	}
	l = m1;
	for (k = 1; k <= n; k++) {
		dum = a[k][1];
		i = k;
		if (l < n) l++;
		for (j = k + 1; j <= l; j++) {
			if (fabs(a[j][1]) > fabs(dum)) {
				dum = a[j][1];
				i = j;
			}
		}
		indx[k] = i;
		if (dum == 0.0) a[k][1] = 1.0e-20;
		if (i != k) {
			for (j = 1; j <= mm; j++) SWAP(REAL, a[k][j], a[i][j])
		}
		for (i = k + 1; i <= l; i++) {
			dum = a[i][1] / a[k][1];
			al[k][i - k] = dum;
			for (j = 2; j <= mm; j++) a[i][j - 1] = a[i][j] - dum*a[k][j];
			a[i][mm] = 0.0;
		}
	}
}

void banbks(MATRIX a, INTEGER m1, INTEGER m2, MATRIX al, IVECTOR indx, VECTOR b, INDEX sts)
{
	INTEGER i, k, l, mm, n = sts.m2;
	REAL dum;

	mm = m1 + m2 + 1;
	l = m1;
	for (k = 1; k <= n; k++) {
		i = indx[k];
		if (i != k) SWAP(REAL, b[k], b[i])
			if (l < n) l++;
		for (i = k + 1; i <= l; i++) b[i] -= al[k][i - k] * b[k];
	}
	l = 1;
	for (i = n; i >= 1; i--) {
		dum = b[i];
		for (k = 2; k <= l; k++) dum -= a[i][k] * b[k + i - 1];
		b[i] = dum / a[i][1];
		if (l < mm) l++;
	}
}

INTEGER ludcmp(MATRIX a, IVECTOR indx, INDEX sts)
{
	INTEGER i, imax=1, j, k;
	REAL big, dum, sum, temp;
	VECTOR vv;

	vv = MakeVector(sts);
	//*d=1.0;
	for (i = sts.n1; i <= sts.n2; i++) {
		big = 0.0;
		for (j = sts.n1; j <= sts.n2; j++)
			if ((temp = fabs(a[i][j])) > big) big = temp;
		if (big == 0.0)
			return(1);
		vv[i] = 1.0 / big;
	}
	for (j = sts.n1; j <= sts.n2; j++) {
		for (i = sts.n1; i<j; i++) {
			sum = a[i][j];
			for (k = sts.n1; k<i; k++) sum -= a[i][k] * a[k][j];
			a[i][j] = sum;
		}
		big = 0.0;
		for (i = j; i <= sts.n2; i++) {
			sum = a[i][j];
			for (k = sts.n1; k<j; k++)
				sum -= a[i][k] * a[k][j];
			a[i][j] = sum;
			if ((dum = vv[i] * fabs(sum)) >= big) {
				big = dum;
				imax = i;
			}
		}
		if (j != imax) {
			for (k = sts.n1; k <= sts.n2; k++) {
				dum = a[imax][k];
				a[imax][k] = a[j][k];
				a[j][k] = dum;
			}
			//*d = -(*d);
			vv[imax] = vv[j];
		}
		indx[j] = imax;
		if (a[j][j] == 0.0) a[j][j] = 1e-20;
		if (j != sts.n2) {
			dum = 1.0 / (a[j][j]);
			for (i = j + 1; i <= sts.n2; i++) a[i][j] *= dum;
		}
	}
	FreeVector(vv, sts);
	return(0);
}

void lubksb(MATRIX a, IVECTOR indx, VECTOR b, INDEX sts)
{
	INTEGER i, ii = 0, ip, j;
	REAL sum;

	for (i = sts.n1; i <= sts.n2; i++) {
		ip = indx[i];
		sum = b[ip];
		b[ip] = b[i];
		if (ii)
			for (j = ii; j <= i - 1; j++) sum -= a[i][j] * b[j];
		else if (sum) ii = i;
		b[i] = sum;
	}
	for (i = sts.n2; i >= sts.n1; i--) {
		sum = b[i];
		for (j = i + 1; j <= sts.n2; j++) sum -= a[i][j] * b[j];
		b[i] = sum / a[i][i];
	}
}

INTEGER EigenRate(VECTOR ev, MATRIX Q, INDEX bnd)
{
	INTEGER i, j, ii, jj, n, m, info;
	INDEX pck;
	VECTOR u, ab, z = NULL;

	n = bnd.m2;
	m = (bnd.n2 - 1) / 2;

	pck.m1 = 0;
	pck.m2 = 1;
	pck.n1 = 0;
	pck.n2 = n*(m + 1);

	u = MakeVector(pck);
	ab = MakeVector(pck);

	for (i = 1; i <= n; i++)
		for (j = 0; j <= m; j++)
		{
			u[(m + 1)*(i - 1) + j] = (m + 1)*(i - 1) + j;
			ii = i - m + j;
			jj = 2 * m + 1 - j;
			if (ii >= 1)
				ab[(m + 1)*(i - 1) + j] = Q[ii][jj];
		}

	//SaveVector("ab.txt", u, ab, pck);

	info = LAPACKE_dsbevd(LAPACK_COL_MAJOR, 'N', 'U', n, m, ab, m + 1, ev + 1, z, 1);

	FreeVector(u, pck);
	FreeVector(ab, pck);
	return info;
}

void Calc_Rate_Pair (MATRIX Q, MATRIX Qsym, VECTOR aq, VECTOR bq, MATRIX A1, INTEGER na1, INTEGER nE1, MATRIX B2, INTEGER na2, INTEGER nE2,
                     VECTOR PqE, IMATRIX state, INTEGER st, VECTOR stq, VECTOR stE,INTEGER m)
{
    INTEGER newst;
    REAL tmpA,tmpB,tmpC,tmpD,H,kT,beta;

    kT = 0.086174*(TdegC+273.15);
    beta = (kT == 0.0)?0.0:1.0/kT;

    tmpA = A1[na1][nE1];
    if (tmpA)
        if ((newst = state[na2][nE2]))	// is new state included in truncated potential?
        {
            H = stE[newst]-stE[st]-(stq[newst]-stq[st])*V;
            tmpA *= ao*exp(-beta*x*H);
            tmpB = B2[na2][nE2]*(ao/Ko)*exp(beta*(1.0-x)*H);		// reverse transition
            tmpC = sqrt(tmpA*tmpB);
            tmpD = sqrt(PqE[newst]/PqE[st]);

            Q[st][newst-st+m+1] = tmpC*tmpD;
            Q[newst][st-newst+m+1] = tmpC/tmpD;

            // diagonals
            Q[st][m+1] -= tmpC*tmpD;
            Q[newst][m+1] -= tmpC / tmpD;

			Qsym[st][newst - st + m + 1] = tmpC;
			Qsym[newst][st - newst + m + 1] = tmpC;

			// diagonals
			Qsym[st][m + 1] -= tmpC*tmpD;
			Qsym[newst][m + 1] -= tmpC / tmpD;

            // projection onto q-axis
            aq[na1] += PqE[st]*tmpC*tmpD;
            bq[na2] += PqE[newst]*tmpC/tmpD;
        }
}

VECTOR MakeVector (INDEX indx)
{
        INTEGER i, n = indx.n2-indx.n1+1;
        VECTOR V;

        V = (VECTOR) malloc( (unsigned)(n) * sizeof(REAL) );
        if (V == NULL) return NULL;

        for(i=0; i<n; i++)
                V[i] = 0.0;

        return V-indx.n1;
}

void FreeVector (VECTOR V, INDEX indx)
{
        free( (CHARACTER*) (V+indx.n1) );
}

IVECTOR MakeIntVector (INDEX indx)
{
        INTEGER i, n = indx.n2-indx.n1+1;
        IVECTOR iV;

        iV = (IVECTOR) malloc( (unsigned)(n) * sizeof(INTEGER) );
        if (iV == NULL) return NULL;

        for(i=0; i<n; i++)
                iV[i] = 0;

        return iV-indx.n1;
}

void FreeIntVector (IVECTOR iV, INDEX indx)
{
        free( (CHARACTER*) (iV+indx.n1) );
}

MATRIX MakeMatrix (INDEX indx)
{
        INTEGER i,j, m = indx.m2-indx.m1+1, n = indx.n2-indx.n1+1;
        MATRIX M;

        M = (MATRIX) malloc( (unsigned)(m) * sizeof(VECTOR) );
        if (M == NULL) return NULL;
        M -= indx.m1;

        M[indx.m1] = (VECTOR) malloc( (unsigned)(m*n*sizeof(REAL)) );
        if (M[indx.m1] == NULL) return NULL;
        M[indx.m1] -= indx.n1;

        for(i=indx.m1; i<indx.m2; i++)
                M[i+1] = M[i]+n;

        for(i=indx.m1; i<=indx.m2; i++)
                for(j=indx.n1; j<=indx.n2; j++)
                        M[i][j] = 0.0;
        return M;
}

void FreeMatrix(MATRIX M, INDEX indx)
{
        free( (CHARACTER*) (M[indx.m1]+indx.n1) );
        free( (CHARACTER*) (M+indx.m1) );
}

IMATRIX MakeIntMatrix (INDEX indx)
{
        INTEGER i,j, m = indx.m2-indx.m1+1, n = indx.n2-indx.n1+1;
        IMATRIX M;

        M = (IMATRIX) malloc( (unsigned)(m) * sizeof(IVECTOR) );
        if (M == NULL) return NULL;
        M -= indx.m1;

        M[indx.m1] = (IVECTOR) malloc( (unsigned)(m*n*sizeof(INTEGER)) );
        if (M[indx.m1] == NULL) return NULL;
        M[indx.m1] -= indx.n1;

        for(i=indx.m1; i<indx.m2; i++)
                M[i+1] = M[i]+n;

        for(i=indx.m1; i<=indx.m2; i++)
                for(j=indx.n1; j<=indx.n2; j++)
                        M[i][j] = 0;
        return M;
}

 void FreeIntMatrix(IMATRIX M, INDEX indx)
{
        free( (CHARACTER*) (M[indx.m1]+indx.n1) );
        free( (CHARACTER*) (M+indx.m1) );
}



#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

long SeedRandomNumber (void)
{
        return -(long) time(NULL)/2;
}

REAL RandomNumber(long *idum)
{
	INTEGER j;
	long k;
	static long idum2 = 123456789;
	static long iy = 0;
	static long iv[NTAB];
	REAL temp;

	if (*idum <= 0)
	{
		if (-(*idum) < 1) *idum = 1;
		else *idum = -(*idum);
		idum2 = (*idum);
		for (j = NTAB + 7; j >= 0; j--)
		{
			k = (*idum) / IQ1;
			*idum = IA1*(*idum - k*IQ1) - k*IR1;
			if (*idum < 0) *idum += IM1;
			if (j < NTAB) iv[j] = *idum;
		}
		iy = iv[0];
	}
	k = (*idum) / IQ1;
	*idum = IA1*(*idum - k*IQ1) - k*IR1;
	if (*idum < 0) *idum += IM1;
	k = idum2 / IQ2;
	idum2 = IA2*(idum2 - k*IQ2) - k*IR2;
	if (idum2 < 0) idum2 += IM2;
	j = iy / NDIV;
	iy = iv[j] - idum2;
	iv[j] = *idum;
	if (iy < 1) iy += IMM1;
	if ((temp = AM*iy) > RNMX) return RNMX;
	else return temp;
}

#undef IM1
#undef IM2
#undef AM
#undef IMM1
#undef IA1
#undef IA2
#undef IQ1
#undef IQ2
#undef IR1
#undef IR2
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX

unsigned GetSeed(void)
{
	INTEGER i, seed1, seed2;

	srand((unsigned)time(NULL));

	for (i = 1; i <= 100; i++)
	{
		seed1 = rand();
		seed2 = rand();
	}
	return (unsigned)seed1 << 16 | seed2;
}

void EigenSystemXDE (MATRIX lEV, MATRIX rEV, VECTOR ev, MATRIX Ap, VECTOR Pe, INDEX tdi)
{
        INTEGER i,j;

        for (i=tdi.n1;i<=tdi.n2;i++)
                lEV[tdi.n1][i]= Ap[tdi.m1+1][i];
        for (i=tdi.n1;i<=tdi.n2;i++)
                lEV[tdi.n1+1][i]= Ap[tdi.m1][i];

        for(i=tdi.n1;i<=tdi.n2;i++)
                for(j=tdi.n1;j<=tdi.n2;j++)
                        if (i==j)
                                rEV[i][i] = 1.0;
                        else
                                rEV[i][j] = 0.0;

        tqli(lEV[tdi.n1],lEV[tdi.n1+1],tdi,rEV);

        for (i=tdi.n1;i<=tdi.n2;i++)
                ev[i] = lEV[tdi.n1][i];

        for (i=tdi.n1;i<=tdi.n2;i++)
                for(j=tdi.n1;j<=tdi.n2;j++)
                        rEV[i][j]/=sqrt(Pe[i]);

        EigSort(tdi,ev,rEV);

        for (i=tdi.n1;i<=tdi.n2;i++)
                for(j=tdi.n1;j<=tdi.n2;j++)
                        lEV[i][j]=rEV[j][i]*Pe[j];
}

void EigSort(INDEX sts, VECTOR ev, MATRIX EV)
{
        INTEGER k,j,i;
        REAL p;

        for (i=sts.n1;i<sts.n2;i++)
        {
                p=ev[k=i];
                for (j=i+1;j<=sts.n2;j++)
                        if (ev[j] >= p) p=ev[k=j];
                if (k != i)
                {
                        ev[k]=ev[i];
                        ev[i]=p;
                        for (j=sts.n1;j<=sts.n2;j++)
                        {
                                p=EV[j][i];
                                EV[j][i]=EV[j][k];
                                EV[j][k]=p;
                        }
                }
        }
}

void EigSort_(VECTOR ev, INDEX sts)
{
	INTEGER k, j, i;
	REAL p;

	for (i = sts.n1; i<sts.n2; i++)
	{
		p = ev[k = i];
		for (j = i + 1; j <= sts.n2; j++)
			if (ev[j] >= p) p = ev[k = j];
		if (k != i)
		{
			ev[k] = ev[i];
			ev[i] = p;
		}
	}
}

void tqli(VECTOR d, VECTOR e, INDEX sts, MATRIX Z)
{
        INTEGER m,l,i,k;
        REAL s,r,p,g,f,dd,c,b;

        for (i=sts.n1+1;i<=sts.n2;i++)
                e[i-1]=e[i];
        e[sts.n2]=0.0;
        for (l=sts.n1;l<=sts.n2;l++)
        {
                do
                {
                        for (m=l;m<sts.n2;m++)
                        {
                                dd=ABS(d[m])+ABS(d[m+1]);
                                if ((ABS(e[m])+dd) == dd) break;
                        }
                        if (m != l)
                        {
                                g=(d[l+1]-d[l])/(2.0*e[l]);
                                r=pythag(g,1.0);
                                g=d[m]-d[l]+e[l]/(g+SIGN(r,g));
                                s=c=1.0;
                                p=0.0;
                                for (i=m-1;i>=l;i--)
                                {
                                        f=s*e[i];
                                        b=c*e[i];
                                        e[i+1]=(r=pythag(f,g));
                                        if (r == 0.0)
                                        {
                                                d[i+1] -= p;
                                                e[m]=0.0;
                                                break;
                                        }
                                        s=f/r;
                                        c=g/r;
                                        g=d[i+1]-p;
                                        r=(d[i]-g)*s+2.0*c*b;
                                        d[i+1]=g+(p=s*r);
                                        g=c*r-b;
                                        for (k=sts.n1;k<=sts.n2;k++)
                                        {
                                                f=Z[k][i+1];
                                                Z[k][i+1]=s*Z[k][i]+c*f;
                                                Z[k][i]=c*Z[k][i]-s*f;
                                        }
                                }
                                if (r == 0.0 && i >= l) continue;
                                d[l] -= p;
                                e[l]=g;
                                e[m]=0.0;
                        }
                } while (m != l);
        }
}

REAL pythag(REAL a, REAL b)
{
        REAL absa,absb;
        absa=ABS(a);
        absb=ABS(b);
        if (absa > absb) return (REAL) absa*sqrt(1.0+SQR(absb/absa));
        else return (REAL) (absb == 0.0 ? 0.0 : absb*sqrt(1.0+SQR(absa/absb)));
}

