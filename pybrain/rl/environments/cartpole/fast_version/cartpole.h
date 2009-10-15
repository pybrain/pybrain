// declaring the methods that can be used from the pyrexwrapper


// CHECKME: are those 'extern's necessary?

extern void initCartPole(int markov_, int numPoles_, int maxsteps_);
extern void reset();
extern unsigned int getObservationDimension();
extern void echoParams();
extern void getObservation(double * input);
extern void doAction(double * output);
extern int trialFinished();
extern double getReward();
