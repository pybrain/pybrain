#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>

using namespace std;

double noise;
int numPoles;
double longPoleAngle;
double poleangle;
bool markov;
double polelength;
double dydx[6];
double state[6];
int episodes;
int totalIterations;
int steps;
int maxsteps;


//////////////////////////////////////////////////////////////////////
//
// Double Pole physics
//
//////////////////////////////////////////////////////////////////////

static const double TRACK_SIZE  = 2.4;
static const double MUP         = 0.000002;
static const double MUC         = 0.0005;
static const double GRAVITY     = -9.8;
static const double MASSCART    = 1.0;
static const double MASSPOLE_1  = 0.1;
double MASSPOLE_2        = 0.01;
static const double LENGTH_1    = 0.5;	 // actually half the pole's length
double LENGTH_2          = 0.05;
static const double FORCE_MAG   = 10.0;  //magnitude of max force
static const double FORCE_MAG2   = 20.0;  //magnitude of max force times 2
static const double TAU         = 0.01;	  //seconds between state updates

#define BIAS 0.01
#define one_degree 0.0174532	/* 2pi/360 */
#define six_degrees 0.1047192
#define twelve_degrees 0.2094384
#define fifteen_degrees 0.2617993
#define thirty_six_degrees 0.628329
#define degrees64    1.2566580
#define fifty_degrees 0.87266

#define BANG_BANG false

#include "cartpole.h"



char modelfile[100];

extern void initCartPole(int markov_, int numPoles_, int maxsteps_)	
{
  poleangle = 4.0156035;
  longPoleAngle = poleangle * one_degree;
  polelength = 0.1;
  LENGTH_2 = polelength/2;
  MASSPOLE_2 = polelength/10;
  noise = 0.0;
  markov = markov_;
  numPoles = numPoles_;
  maxsteps = maxsteps_;

  reset();

  episodes = 0;
  totalIterations = 0;


  //echoParams();
}



extern unsigned int getObservationDimension()
{
  unsigned int inputDimension = 0;

  switch (numPoles){
  case 1:
    if(!markov) {         //**********************
      inputDimension = 3; // BIAS
                          //**********************
    }
    else inputDimension = 5;
    break;
  case 2:
    if(!markov) {
      inputDimension = 4;
    }
    else inputDimension = 7;
    break;
  };

  return inputDimension;
}


/*
void echoParams(){
  cout << "\nCart-pole environment settings: \n";
  cout << "------------------------------\n";
  cout << "Number of poles            : " << numPoles << "\n";
  cout << "Length of short pole       : " << LENGTH_2 * 2 << " meters\n";
  cout << "Initial angle of long pole : " << longPoleAngle/one_degree << " degrees\n";
  cout << "Number of inputs           : " << getObservationDimension() << "\n";
  if(markov)
    cout << "Markov -- full state information." << endl;
  else
    cout << "Non-Markov -- no velocity information." << endl;
  if(noise)
    cout << "Percent sensor noise    : " << noise * 50 << "\n";
  if(BANG_BANG)
    cout << "BANG BANG control" << endl;
}
*/

extern void reset()
{
  dydx[0] = dydx[1] = dydx[2] = dydx[3] =  dydx[4] = dydx[5] = 0.0;
  state[0] = state[1] = state[3] = state[4] = state[5] = 0;
  state[2] = longPoleAngle;
  steps = 0;
  episodes++;
}







#define one_over_256  0.0390625
void step(double action, double *st, double *derivs)
{
  double force;
  double costheta_1, costheta_2=0.0;
  double sintheta_1, sintheta_2=0.0;
  double gsintheta_1,gsintheta_2=0.0;
  double temp_1,temp_2=0.0;
  double ml_1, ml_2;
  double fi_1,fi_2=0.0;
  double mi_1,mi_2=0.0;

  if(BANG_BANG){
    if(action > 0.5) force = FORCE_MAG;
    else force = -FORCE_MAG;
  }
  else
    force =  (action /*- 0.5*/) * FORCE_MAG;
    //force =  (action - 0.5) * FORCE_MAG2;
  if(force > FORCE_MAG)
    force = FORCE_MAG;
  if(force < -FORCE_MAG)
    force = -FORCE_MAG;


  if((force >= 0) && (force < one_over_256))
    force = one_over_256;
  if((force < 0) && (force > -one_over_256))
    force = -one_over_256;


  costheta_1 = cos(st[2]);
  sintheta_1 = sin(st[2]);
  gsintheta_1 = GRAVITY * sintheta_1;
  ml_1 = LENGTH_1 * MASSPOLE_1;
  temp_1 = MUP * st[3] / ml_1;
  fi_1 = (ml_1 * st[3] * st[3] * sintheta_1) +
    (0.75 * MASSPOLE_1 * costheta_1 * (temp_1 + gsintheta_1));
  mi_1 = MASSPOLE_1 * (1 - (0.75 * costheta_1 * costheta_1));

  if(numPoles > 1){
    costheta_2 = cos(st[4]);
    sintheta_2 = sin(st[4]);
    gsintheta_2 = GRAVITY * sintheta_2;
    ml_2 = LENGTH_2 * MASSPOLE_2;
    temp_2 = MUP * st[5] / ml_2;
    fi_2 = (ml_2 * st[5] * st[5] * sintheta_2) +
      (0.75 * MASSPOLE_2 * costheta_2 * (temp_2 + gsintheta_2));
    mi_2 = MASSPOLE_2 * (1 - (0.75 * costheta_2 * costheta_2));
  }

  derivs[1] = (force + fi_1 + fi_2)
	/ (mi_1 + mi_2 + MASSCART);

  derivs[3] = -0.75 * (derivs[1] * costheta_1 + gsintheta_1 + temp_1)
    / LENGTH_1;
  if(numPoles > 1)
    derivs[5] = -0.75 * (derivs[1] * costheta_2 + gsintheta_2 + temp_2)
      / LENGTH_2;

}

void rk4(double f, double y[], double dydx[], double yout[])
{

	int i;

	double hh,h6,dym[6],dyt[6],yt[6];
	int vars = 3;

	if(numPoles > 1)
	  vars = 5;

	hh=TAU*0.5;
	h6=TAU/6.0;
	for (i=0;i<=vars;i++) yt[i]=y[i]+hh*dydx[i];
	step(f,yt,dyt);
	dyt[0] = yt[1];
	dyt[2] = yt[3];
	dyt[4] = yt[5];
	for (i=0;i<=vars;i++) yt[i]=y[i]+hh*dyt[i];
	step(f,yt,dym);
	dym[0] = yt[1];
	dym[2] = yt[3];
	dym[4] = yt[5];
	for (i=0;i<=vars;i++) {
	  yt[i]=y[i]+TAU*dym[i];
	  dym[i] += dyt[i];
	}
	step(f,yt,dyt);
	dyt[0] = yt[1];
	dyt[2] = yt[3];
	dyt[4] = yt[5];
	for (i=0;i<=vars;i++)
	  yout[i]=y[i]+h6*(dydx[i]+dyt[i]+2.0*dym[i]);
}



extern void getObservation(double * input)
{
  // first is always bias, last is always cart position (state[0])
  // pole angles are preceded by their dreivative in the markov case
  switch(numPoles){
    case 1:
      if(markov){
	input[4] = (state[0] / 2.4-0.01)*10                + (rand() * noise - (noise/2));
	input[1] = (state[1] / 10.0-0.01)*10               + (rand() * noise - (noise/2));
	input[2] = state[2] / twelve_degrees     + (rand() * noise - (noise/2));
	input[3] = 10*state[3] / 5.0                + (rand() * noise - (noise/2));
	input[0] = BIAS;
      }
      else{
	input[2] = 10* (state[0] / 2.4-0.01)                + (rand() * noise - (noise/2));
	input[1] = state[2] / twelve_degrees     + (rand() * noise - (noise/2));
	input[0] = BIAS;
      }
      break;
    case 2:
      if(markov){
	input[6] = 10*(state[0] / 2.4-0.01)                + (rand() * noise - (noise/2));
	input[1] = 10*(state[1] / 10.0-0.01)               + (rand() * noise - (noise/2));
	input[2] = state[2] / twelve_degrees;//thirty_six_degrees + (rand() * noise - (noise/2));
	input[3] = 10*state[3] / 5.0                + (rand() * noise - (noise/2));
	input[4] = state[4] / thirty_six_degrees + (rand() * noise - (noise/2));
	input[5] = state[5] / 16.0               + (rand() * noise - (noise/2));
	input[0] = BIAS;
      }
      else{

	input[3] = 10*(state[0] / 2.4-0.01)                + (rand() * noise - (noise/2));
	input[1] = state[2] / twelve_degrees               + (rand() * noise - (noise/2));
	input[2] = state[4] / 0.52               + (rand() * noise - (noise/2));
	input[0] = BIAS;


      }
      break;
    };
}

#define RK4 1
#define EULER_TAU (TAU/8)
extern void doAction(double * output)
{

  int i;
  double tmpState[6];
  double force;

  force = output[0];
  /*random start state for long pole*/
  /*state[2]= rand();   */


  /*--- Apply action to the simulated cart-pole ---*/

  if(RK4){
    dydx[0] = state[1];
    dydx[2] = state[3];
    dydx[4] = state[5];
    step(force,state,dydx);
    rk4(force,state,dydx,state);
    for(i=0;i<6;++i)
      tmpState[i] = state[i];
    dydx[0] = state[1];
    dydx[2] = state[3];
    dydx[4] = state[5];
    step(force,state,dydx);
    rk4(force,state,dydx,state);
  }
  else{
    for(i=0;i<16;++i){
      step(output[0],state,dydx);
      state[0] += EULER_TAU * state[1];
      state[1] += EULER_TAU * dydx[1];
      state[2] += EULER_TAU * state[3];
      state[3] += EULER_TAU * dydx[3];
      state[4] += EULER_TAU * state[5];
      state[5] += EULER_TAU * dydx[5];
    }
  }

  steps++;
  totalIterations++;
}


bool outsideBounds()
{
  double failureAngle;

  if(numPoles > 1){
    failureAngle = thirty_six_degrees;
    return
      fabs(state[0]) > TRACK_SIZE       ||
      fabs(state[2]) > failureAngle     ||
      fabs(state[4]) > failureAngle;
  }
  else{
    failureAngle = twelve_degrees;
    return
      fabs(state[0]) > TRACK_SIZE       ||
      fabs(state[2]) > failureAngle;
  }
}



extern int trialFinished()
{
  if(outsideBounds())
    return 1;
  if(steps > maxsteps) {
    //cout << "SUCCESS in episode " << episodes << endl;
    return 1;
  }
  return 0;
}



double getReward()
{
  if(outsideBounds())
    return -1.0;
  return 0.0;
}


