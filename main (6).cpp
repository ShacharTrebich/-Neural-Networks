/***********************************************************************
*                       Neural Networks 							   *
*               Project number 2 - Back Propagation					   *
*                   												   *
************************************************************************
*/

//This is a Back Propagation network. It consists layers:
//25 neurons on input layer, 10 neurons on hidden layer and one
//neuron on output layer.

//This programm use function: F(NET) = tanh (NET), that takes values
//from -1 to +1.
//The values of the neurons in the hidden layer are continuous.
//The values of the input and output neurons are diskreet.

//This program do not print anything to display, and all results of
//this programm will be in file "result.txt" and "bias_result.txt"
//after runing of this programm.
//Before runing programm againe, please delete old file with results.





#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
// #include <io.h>   IF WE NEED IT


using namespace std;

#include "Numbers.dat"        //File with patterns for input end output.

#define Low           0
#define Hi	          +1
#define Bias          1

#define InputNeurons  48
#define HiddenNeurons1 50
#define HiddenNeurons2_1 100  // WE CAN CHOSE ANY NUMBER THAT WE WANT
#define HiddenNeurons2_2 80
#define HiddenNeurons2_3 60
#define OutputNeurons 4


#define sqr(x) ((x)*(x)) // square

//typedef int InArr[InputNeurons];
typedef int OutArr[OutputNeurons];

class Data
{
	public:
	int Input[10][48];
	int Output[10][4];
	int Units;     //Numbers (units) in input ( and output ) now.
	Data ();
	~Data();
	   //Set input and output vectors from patterns.
	bool SetInputOutput(char[][y][x], char[][5], int);
       //Free memory of Input and Output units
	void Reset();
};

class BackPropagationNet
{
	private:
	    int    InputLayer[InputNeurons];
		//Output from hidden layer -> it is input to output layer.
		float  HiddenLayer1[HiddenNeurons1];
		//Output of network - one neuron.
		float  OutputLayer[OutputNeurons];	                              					   // THIS IS RESULT OF NETWORK RUNNING
        float  WeigthsOut[OutputNeurons][HiddenNeurons1];
		float  WeigthsHidd1[HiddenNeurons1][InputNeurons];

	    //the arrays for 3 hidden layers
	    float  HiddenLayer2_1[HiddenNeurons2_1];
		float  HiddenLayer2_2[HiddenNeurons2_2];
		float  HiddenLayer2_3[HiddenNeurons2_3];
		float  WeigthsHidd2_1[HiddenNeurons2_1][InputNeurons];
		float  WeigthsHidd2_2[HiddenNeurons2_2][HiddenNeurons2_1];
		float  WeigthsHidd2_3[HiddenNeurons2_3][HiddenNeurons2_2];
	    float  WeigthsOut_2[OutputNeurons][HiddenNeurons2_3];
	    float  nu;  // leraning rate                                    									  //Learning rate.
		float  Threshold;       // for tounding weights in output
	    //It was error now ?. If error occured, then NetError = true,
		//else NetError = false.
	    bool  NetError;
		float RandomEqualReal(float, float);
		//Calculate output for current input (without Bias).
		void CalculateOutput();
		void CalculateOutput_3_hidden();
		void ItIsError(int);           //NetError = true if it was error.
		void AdjustWeigths(int []);
		void AdjustWeigths_3_hidden(int []);
	public:
        //Initialization of weigths and variables.
	    BackPropagationNet();
		//Initialize all and randomly weigths.
	    void  Initialize();
		//Train network up to 90% success or up to 20000 cycles
		bool TrainNet(Data &);
		bool TrainNet_2_groups(Data &, Data &);
		bool TrainRandomNet(Data &);
		bool TrainRandomNet_2_groups(Data &, Data &);

		bool TrainNet_3_hidden(Data &);
		bool TrainNet_2_groups_3_hidden(Data &, Data &);
		bool TrainRandomNet_3_hidden(Data &);
		bool TrainRandomNet_2_groups_3_hidden(Data &, Data &);
        //Testing of network. Return success percent.
		int TestNet(Data &);
		int TestNet_3_hidden(Data &);
	    int ReturnOutput();
	    float LearningRate()        { return nu; };
	    float ThresholdValue()      { return Threshold; };
};


BackPropagationNet::BackPropagationNet()                                        // LEARNING RATE
{
    nu = 0.1f;                      // we should check few different learning rates and find the one that will work the best

	srand(time(0));
    Initialize();
}

void BackPropagationNet::Initialize()                                          // INITIALIZATION
{
    int i, j;

	Threshold = 0.5f;   // we could change treshold every time we found a mistake
    NetError = false;   // or let it be constant
    //Randomize weigths (initialize).
	for(i = 0; i < OutputNeurons; i++)
	{
		for(int j = 0; j < HiddenNeurons1; j++)
		{
			WeigthsOut[i][j] = RandomEqualReal(-1.0f, 1.0f);
		}
	}
	//Randomize weigths for 1 hidden layer

	for(i = 0; i < HiddenNeurons1; i++)
    {
	    for(j = 0; j < InputNeurons; j++)
	    {
	        WeigthsHidd1[i][j] = RandomEqualReal(-1.0f, 1.0f);
		}
    }

    // Randomize weights_out for 3 hidden layer
    for(i = 0; i < OutputNeurons; i++)
	{
		for(int j = 0; j < HiddenNeurons2_3; j++)
		{
			WeigthsOut_2[i][j] = RandomEqualReal(-1.0f, 1.0f);
		}
	}
	//Randomize weigths for 3 hidden layers
	for(i = 0; i < HiddenNeurons2_1; i++)
    {
	    for(j = 0; j < InputNeurons; j++)
	    {
	        WeigthsHidd2_1[i][j] = RandomEqualReal(-1.0f, 1.0f);
		}
    }
	for(i = 0; i < HiddenNeurons2_2; i++)
    {
	    for(j = 0; j < HiddenNeurons2_1; j++)
	    {
	        WeigthsHidd2_2[i][j] = RandomEqualReal(-1.0f, 1.0f);
		}
    }
    for(i = 0; i < HiddenNeurons2_3; i++)
    {
	    for(j = 0; j < HiddenNeurons2_2; j++)
	    {
	        WeigthsHidd2_3[i][j] = RandomEqualReal(-1.0f, 1.0f);
		}
    }
}


//Return randomaly numbers from LowN to HighN
float BackPropagationNet::RandomEqualReal(float LowN, float HighN)             // RANDOM NUMBER BETWEEN [LOW - HIGH]
{
    return (((float) rand()) / (float)RAND_MAX) * (HighN - LowN) + LowN;
}

void BackPropagationNet::CalculateOutput()                              // CALCULATE OUTPUT OF OUR SYSTEM
{
    float Sum;                                                         // WITH 1 HIDDEN LAYER
                                                                     // WE NEED TO BUILD IT WITH 3 HIDDEN LAYERS
	//Calculate output for hidden layer.
    for(int i = 0; i < HiddenNeurons1; i++)
    {
	    Sum = 0.0f; 						// sum from the input into first hidden layer
	    for(int j = 0; j < InputNeurons; j++)
	    {
	        Sum += WeigthsHidd1[i][j] * InputLayer[j];
	    }
		HiddenLayer1[i] = (float)(1/ (1 + exp(-Sum)));   // we use the function tanh to convert numbers into [-1; 1] interval
    }
	//Calculate output for output layer.

	for(int i = 0; i < DataOutputs; i++)
	{
		Sum = 0.0f;
		for(int j = 0; j < HiddenNeurons1; j++)
		{
				Sum += WeigthsOut[i][j] * HiddenLayer1[j];
		}
		OutputLayer[i] = (float)(1/ (1 + exp(-Sum)));
	}
}

void BackPropagationNet::CalculateOutput_3_hidden()                              // CALCULATE OUTPUT FOR 3 HIDD LAYERS OF OUR SYSTEM
{
    float Sum;
	//Calculate output for first hidden layer.
    for(int i = 0; i < HiddenNeurons2_1; i++)
    {
	    Sum = 0.0f; 						// sum from the input into first hidden layer
	    for(int j = 0; j < InputNeurons; j++)
	    {
	        Sum += WeigthsHidd2_1[i][j] * InputLayer[j];
	    }
		HiddenLayer2_1[i] = (float)(1/ (1 + exp(-Sum)));
    }
	//Calculate output for second hidden layer.
	for(int i = 0; i < HiddenNeurons2_2; i++)
    {
	    Sum = 0.0f;
	    for(int j = 0; j < HiddenNeurons2_1; j++)
	    {
	        Sum += WeigthsHidd2_2[i][j] * HiddenLayer2_1[j];
	    }
		HiddenLayer2_2[i] = (float)(1/ (1 + exp(-Sum)));
    }
	//Calculate output for third hidden layer.
	for(int i = 0; i < HiddenNeurons2_3; i++)
    {
	    Sum = 0.0f;
	    for(int j = 0; j < HiddenNeurons2_2; j++)
	    {
	        Sum += WeigthsHidd2_3[i][j] * HiddenLayer2_2[j];
	    }
		HiddenLayer2_3[i] = (float)(1/ (1 + exp(-Sum)));
    }
	//Calculate output layer
	for(int i = 0; i < DataOutputs; i++)
	{
		Sum = 0.0f;
		for(int j = 0; j < HiddenNeurons2_3; j++)
		{
			Sum += WeigthsOut_2[i][j] * HiddenLayer2_3[j];
		}
		OutputLayer[i] = (float)(1/ (1 + exp(-Sum))); // we get output layer
	}
}

void BackPropagationNet::ItIsError(int Target)            //  CHECK IF THERE IS AN ERROR IN OUTPUT
{
	int answ = 0;
	if(ReturnOutput() != Target)
	{
		NetError = true;
	}
	else
		NetError = false;
}


void BackPropagationNet::AdjustWeigths(int Target[])          // ADJUST WEIGHTS
{
    int i, j;
    float hidd_deltas[HiddenNeurons1], out_delta[OutputNeurons];
	//Calculate deltas for all layers.
	for(i = 0; i < OutputNeurons; i++)
	{
		out_delta[i] = (float)((1 - sqr(OutputLayer[i])) * ((float)(Target[i] - OutputLayer[i])));
	}
	for(i = 0; i < HiddenNeurons1; i++)
	{
		hidd_deltas[i] = 0.0;
		for(j = 0; j < OutputNeurons; j++)    // loop
		{
			hidd_deltas[i] = hidd_deltas[i] + (float)((1 - sqr(HiddenLayer1[i])) * out_delta[j] * WeigthsOut[i][j]);
		}
	}
	//Change weigths.
	for(j = 0; j < OutputNeurons; j++)
	{
		for(i = 0; i < HiddenNeurons1; i++)
		{
			WeigthsOut[j][i] = WeigthsOut[j][i] + (nu * out_delta[j] * HiddenLayer1[i]);
		}
	}
	for(i = 0; i < HiddenNeurons1; i++)
	{
		for(j = 0; j < InputNeurons; j++)
		{
			WeigthsHidd1[i][j] = WeigthsHidd1[i][j] + (nu * hidd_deltas[i] * InputLayer[j]);
		}
	}
}


void BackPropagationNet::AdjustWeigths_3_hidden(int Target[])          // ADJUST WEIGHTS
{
    int i, j;
    float hidd_deltas1[HiddenNeurons2_1], hidd_deltas2[HiddenNeurons2_2], hidd_deltas3[HiddenNeurons2_3], out_delta[OutputNeurons];
	//Calculate deltas for all layers.
	for(i = 0; i < OutputNeurons; i++)
	{
		out_delta[i] = (float)((1 - sqr(OutputLayer[i])) * ((float)(Target[i] - OutputLayer[i])));
	}
	for(i = 0; i < HiddenNeurons2_3; i++)
	{
		hidd_deltas3[i] = 0.0;
		for(j = 0; j < OutputNeurons; j++)    // loop
		{
			hidd_deltas3[i] = hidd_deltas3[i] + (float)((1 - sqr(HiddenLayer2_3[i])) * out_delta[j] * WeigthsOut_2[j][i]);
		}
	}
	for(i = 0; i < HiddenNeurons2_2; i++)
	{
		hidd_deltas2[i] = 0.0;
		for(j = 0; j < HiddenNeurons2_3; j++)    // loop
		{
			hidd_deltas2[i] = hidd_deltas2[i] + (float)((1 - sqr(HiddenLayer2_2[i])) * hidd_deltas3[j] * WeigthsHidd2_3[j][i]);
		}
	}
	for(i = 0; i < HiddenNeurons2_1; i++)
	{
		hidd_deltas1[i] = 0.0;
		for(j = 0; j < InputNeurons; j++)    // loop
		{
			hidd_deltas1[i] = hidd_deltas1[i] + (float)((1 - sqr(HiddenLayer2_1[i])) * hidd_deltas2[j] * WeigthsHidd2_2[j][i]);
		}
	}
	//Change weigths.
	for(j = 0; j < OutputNeurons; j++)
	{
		for(i = 0; i < HiddenNeurons2_3; i++)
		{
			WeigthsOut_2[j][i] = WeigthsOut_2[j][i] + (nu * out_delta[j] * HiddenLayer2_3[i]);
		}
	}
	for(i = 0; i < HiddenNeurons2_3; i++)
	{
		for(j = 0; j < HiddenNeurons2_2; j++)
		{
			WeigthsHidd2_3[i][j] = WeigthsHidd2_3[i][j] + (nu * hidd_deltas3[i] * HiddenLayer2_2[j]);
		}
	}
	for(i = 0; i < HiddenNeurons2_2; i++)
	{
		for(j = 0; j < HiddenNeurons2_1
		; j++)
		{
			WeigthsHidd2_2[i][j] = WeigthsHidd2_2[i][j] + (nu * hidd_deltas2[i] * HiddenLayer2_1[j]);
		}
	}
	for(i = 0; i < HiddenNeurons2_1; i++)
	{
		for(j = 0; j < InputNeurons; j++)
		{
			WeigthsHidd2_1[i][j] = WeigthsHidd2_1[i][j] + (nu * hidd_deltas1[i] * InputLayer[j]);
		}
	}
}


int BackPropagationNet::ReturnOutput()
{
	int answ = 0, temp = 8;
	for(int i = 0; i < DataOutputs; i++)
	{
		if(OutputLayer[i] > Threshold)
		{
			answ += temp;
		}
		temp /= 2;
	}
	return answ;
}

			// 1 HIDDEN LAYER ALGORITHMS

bool BackPropagationNet::TrainNet( Data & data_obj )      // Training of network
{
    int Error, j, loop = 0, Success;
	cout << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
	cout << "                       TRAINING NETWORK" << endl << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
    do
    {
	    Error = 0;
	    loop ++;
		//Printing the number of loop.
		if( loop < 10 )
            cout <<"Training loop:  " << loop << "       ...   ";
		if( loop >= 10  && loop < 100)
		    cout <<"Training loop:  " << loop << "      ...   ";
		if( loop >= 100 && loop < 1000)
		    cout <<"Training loop:  " << loop << "     ...   ";
		if(	loop >= 1000 && loop < 10000)
			cout <<"Training loop:  " << loop << "    ...   ";
		else if( loop >= 10000 )
			cout << "Training loop:  " << loop << "   ...   ";
        //Train network (do one cycle).
        for(int i = 0; i < data_obj.Units; i++)   // units = 10
        {
	        //Set current input.

	        for(j = 0; j < InputNeurons; j++)
	        {
	            InputLayer[j] = data_obj.Input[i][j];
			}
	        CalculateOutput();
	        int answ_temp = 0, temp = 8;					// This calculate the right outoput and send it to function that compare the network output with the right answer
			for(int k = 0; k < OutputNeurons; k++)
			{
				if(data_obj.Output[i][k] == 1)
				{
					answ_temp += temp;
				}
				temp /= 2;
			}
	        ItIsError(answ_temp);						// this is the comparison
	        //If it was error, change weigths (Error = sum of errors in
			//one cycle of train).
	        if(NetError)
	        {
				Error ++;															// convert integer 0-1 values of data class into float values 0-1
				AdjustWeigths(data_obj.Output[i]);//make an apropriate array sending of func   // WHEN WE ADD DATA FUNCTIONS CHECK THIS FUNCTION THAT IT WILL WORK CORRECT
            }																				// IT USES DATA CLASS FUNCTIONS THAT WE HAVE NOT ADDED YET
	    }
	Success = ((data_obj.Units - Error)*100) / data_obj.Units;
	cout << Success <<" %   success" << endl << endl;
	//cout << WeigthsOut[0][0] << "    " << WeigthsOut[0][1] << "     " << WeigthsOut[0][2]<< endl;               // FOR CHECKING
    }while(Success < 90 && loop <= 20000);									// ITS NOT 20000 FOR PURPOSE OF CHECKING

	if(loop > 20000)
	{
        cout << "Training of network failure !" << endl;
        return false;
	}
	else
		return true;
}


bool BackPropagationNet::TrainNet_2_groups( Data & data_obj_1, Data & data_obj_2)      // Training of network
{
    int Error, j, loop = 0, Success;
	cout << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
	cout << "                       TRAINING NETWORK" << endl << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
    do
    {
	    Error = 0;
	    loop ++;
		//Printing the number of loop.
		if( loop < 10 )
            cout <<"Training loop:  " << loop << "       ...   ";
		if( loop >= 10  && loop < 100)
		    cout <<"Training loop:  " << loop << "      ...   ";
		if( loop >= 100 && loop < 1000)
		    cout <<"Training loop:  " << loop << "     ...   ";
		if(	loop >= 1000 && loop < 10000)
			cout <<"Training loop:  " << loop << "    ...   ";
		else if( loop >= 10000 )
			cout << "Training loop:  " << loop << "   ...   ";
        //Train network (do one cycle).
        for(int i = 0; i < data_obj_1.Units; i++)   // units = 10
        {
	        //Set current input.

	        for(j = 0; j < InputNeurons; j++)
	        {
	            InputLayer[j] = data_obj_1.Input[i][j];
			}
	        CalculateOutput();
	        int answ_temp = 0, temp = 8;					// This calculate the right outoput and send it to function that compare the network output with the right answer
			for(int k = 0; k < OutputNeurons; k++)
			{
				if(data_obj_1.Output[i][k] == 1)
				{
					answ_temp += temp;
				}
				temp /= 2;
			}
	        ItIsError(answ_temp);						// this is the comparison
	        //If it was error, change weigths (Error = sum of errors in
			//one cycle of train).
	        if(NetError)
	        {
				Error ++;															// convert integer 0-1 values of data class into float values 0-1
				AdjustWeigths(data_obj_1.Output[i]);//make an apropriate array sending of func   // WHEN WE ADD DATA FUNCTIONS CHECK THIS FUNCTION THAT IT WILL WORK CORRECT
            }																				// IT USES DATA CLASS FUNCTIONS THAT WE HAVE NOT ADDED YET
	    }
	    for(int i = 0; i < data_obj_2.Units; i++)   // units = 10
        {
	        //Set current input.

	        for(j = 0; j < InputNeurons; j++)
	        {
	            InputLayer[j] = data_obj_2.Input[i][j];
			}
	        CalculateOutput();
	        int answ_temp = 0, temp = 8;					// This calculate the right outoput and send it to function that compare the network output with the right answer
			for(int k = 0; k < OutputNeurons; k++)
			{
				if(data_obj_2.Output[i][k] == 1)
				{
					answ_temp += temp;
				}
				temp /= 2;
			}
	        ItIsError(answ_temp);						// this is the comparison
	        //If it was error, change weigths (Error = sum of errors in
			//one cycle of train).
	        if(NetError)
	        {
				Error ++;															// convert integer 0-1 values of data class into float values 0-1
				AdjustWeigths(data_obj_2.Output[i]);//make an apropriate array sending of func   // WHEN WE ADD DATA FUNCTIONS CHECK THIS FUNCTION THAT IT WILL WORK CORRECT
            }																				// IT USES DATA CLASS FUNCTIONS THAT WE HAVE NOT ADDED YET
	    }
	Success = ((data_obj_1.Units + data_obj_2.Units - Error)*100) / (data_obj_1.Units + data_obj_2.Units);
	cout << Success <<" %   success" << endl << endl;
	//cout << WeigthsOut[0][0] << "    " << WeigthsOut[0][1] << "     " << WeigthsOut[0][2]<< endl;               // FOR CHECKING
    }while(Success < 90 && loop <= 20000);									// ITS NOT 20000 FOR PURPOSE OF CHECKING

	if(loop > 20000)
	{
        cout << "Training of network failure !" << endl;
        return false;
	}
	else
		return true;
}

bool BackPropagationNet::TrainRandomNet( Data & data_obj )      // Training of network
{
    int Error, j, loop = 0, Success;
	cout << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
	cout << "                       TRAINING NETWORK" << endl << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
    do
    {
	    Error = 0;
	    loop ++;
		//Printing the number of loop.
		if( loop < 10 )
            cout <<"Training loop:  " << loop << "       ...   ";
		if( loop >= 10  && loop < 100)
		    cout <<"Training loop:  " << loop << "      ...   ";
		if( loop >= 100 && loop < 1000)
		    cout <<"Training loop:  " << loop << "     ...   ";
		if(	loop >= 1000 && loop < 10000)
			cout <<"Training loop:  " << loop << "    ...   ";
		else if( loop >= 10000 )
			cout << "Training loop:  " << loop << "   ...   ";
        //Train network (do one cycle).
        for(int i = 0; i < data_obj.Units; i++)   // units = 10
        {
	        //Set current input.
	        int arr_rand[Numbers];    // ARRAY WITH RANDOM ORDER OF NUMBERS FROM 0 TO 9
			int h = 0;
			while(h < Numbers)      // MAKE THIS ARRAY
			{
				int rand_help = rand()%10;
				bool already_in = false;
				for(int h2 = 0; h2 < h; h2++)
				{
					if(rand_help == arr_rand[h2])
					{
						already_in = true;
					}
				}
				if(!already_in)
				{
					arr_rand[h] = rand_help;
					h++;
				}
			}
	        for(j = 0; j < InputNeurons; j++)
	        {
	            InputLayer[j] = data_obj.Input[arr_rand[i]][j];
			}
	        CalculateOutput();

	        int answ_temp = 0, temp = 8;					// This calculate the right outoput and send it to function that compare the network output with the right answer
			for(int k = 0; k < sizeof(data_obj.Output[arr_rand[i]]); k++)
			{
				if(data_obj.Output[arr_rand[i]][k] == 1)
				{
					answ_temp += temp;
				}
				temp /= 2;
			}
	        ItIsError(answ_temp);						// this is the comparison

	        //If it was error, change weigths (Error = sum of errors in
			//one cycle of train).
	        if(NetError)
	        {
				Error ++;															// convert integer 0-1 values of data class into float values 0-1
				AdjustWeigths(data_obj.Output[arr_rand[i]]);//make an apropriate array sending of func   // WHEN WE ADD DATA FUNCTIONS CHECK THIS FUNCTION THAT IT WILL WORK CORRECT
            }																				// IT USES DATA CLASS FUNCTIONS THAT WE HAVE NOT ADDED YET
	    }

	Success = ((data_obj.Units - Error)*100) / data_obj.Units;
	cout << Success <<" %   success" << endl << endl;

	//cout << WeigthsOut[0][0] << "    " << WeigthsOut[0][1] << "     " << WeigthsOut[0][2]<< endl;               // FOR CHECKING
    }while(Success < 90 && loop <= 20000);									// ITS NOT 20000 FOR PURPOSE OF CHECKING

	if(loop > 20000)
	{
        cout << "Training of network failure !" << endl;
        return false;
	}
	else
		return true;
}

bool BackPropagationNet::TrainRandomNet_2_groups( Data & data_obj_1, Data & data_obj_2)      // Training of network
{
    int Error, j, loop = 0, Success;
	cout << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
	cout << "                       TRAINING NETWORK" << endl << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
    do
    {
	    Error = 0;
	    loop ++;
		//Printing the number of loop.
		if( loop < 10 )
            cout <<"Training loop:  " << loop << "       ...   ";
		if( loop >= 10  && loop < 100)
		    cout <<"Training loop:  " << loop << "      ...   ";
		if( loop >= 100 && loop < 1000)
		    cout <<"Training loop:  " << loop << "     ...   ";
		if(	loop >= 1000 && loop < 10000)
			cout <<"Training loop:  " << loop << "    ...   ";
		else if( loop >= 10000 )
			cout << "Training loop:  " << loop << "   ...   ";
        //Train network (do one cycle).
        for(int i = 0; i < data_obj_1.Units; i++)   // units = 10
        {
	        //Set current input.
	        int arr_rand[Numbers];    // ARRAY WITH RANDOM ORDER OF NUMBERS FROM 0 TO 9
			int h = 0;
			while(h < Numbers)      // MAKE THIS ARRAY
			{
				int rand_help = rand()%10;
				bool already_in = false;
				for(int h2 = 0; h2 < h; h2++)
				{
					if(rand_help == arr_rand[h2])
					{
						already_in = true;
					}
				}
				if(!already_in)
				{
					arr_rand[h] = rand_help;
					h++;
				}
			}
	        for(j = 0; j < InputNeurons; j++)
	        {
	            InputLayer[j] = data_obj_1.Input[arr_rand[i]][j];
			}
	        CalculateOutput();
	        int answ_temp = 0, temp = 8;					// This calculate the right outoput and send it to function that compare the network output with the right answer
			for(int k = 0; k < OutputNeurons; k++)
			{
				if(data_obj_1.Output[arr_rand[i]][k] == 1)
				{
					answ_temp += temp;
				}
				temp /= 2;
			}
	        ItIsError(answ_temp);						// this is the comparison
	        //If it was error, change weigths (Error = sum of errors in
			//one cycle of train).
	        if(NetError)
	        {
				Error ++;															// convert integer 0-1 values of data class into float values 0-1
				AdjustWeigths(data_obj_1.Output[arr_rand[i]]);//make an apropriate array sending of func   // WHEN WE ADD DATA FUNCTIONS CHECK THIS FUNCTION THAT IT WILL WORK CORRECT
            }																				// IT USES DATA CLASS FUNCTIONS THAT WE HAVE NOT ADDED YET
	    }

	    for(int i = 0; i < data_obj_2.Units; i++)   // units = 10
        {
	        //Set current input for second group.
	        int arr_rand[Numbers];    // ARRAY WITH RANDOM ORDER OF NUMBERS FROM 0 TO 9
			int h = 0;
	        while(h < Numbers)      // MAKE THIS ARRAY RANDOM FOR SECOND GROUP
			{
				int rand_help = rand()%10;
				bool already_in = false;
				for(int h2 = 0; h2 < h; h2++)
				{
					if(rand_help == arr_rand[h2])
					{
						already_in = true;
					}
				}
				if(!already_in)
				{
					arr_rand[h] = rand_help;
					h++;
				}
			}
	        for(j = 0; j < InputNeurons; j++)
	        {
	            InputLayer[j] = data_obj_2.Input[arr_rand[i]][j];
			}
	        CalculateOutput();
	        int answ_temp = 0, temp = 8;					// This calculate the right outoput and send it to function that compare the network output with the right answer
			for(int k = 0; k < OutputNeurons; k++)
			{
				if(data_obj_2.Output[arr_rand[i]][k] == 1)
				{
					answ_temp += temp;
				}
				temp /= 2;
			}
	        ItIsError(answ_temp);						// this is the comparison
	        //If it was error, change weigths (Error = sum of errors in
			//one cycle of train).
	        if(NetError)
	        {
				Error ++;															// convert integer 0-1 values of data class into float values 0-1
				AdjustWeigths(data_obj_2.Output[arr_rand[i]]);//make an apropriate array sending of func   // WHEN WE ADD DATA FUNCTIONS CHECK THIS FUNCTION THAT IT WILL WORK CORRECT
            }																				// IT USES DATA CLASS FUNCTIONS THAT WE HAVE NOT ADDED YET
	    }
	Success = ((data_obj_1.Units + data_obj_2.Units - Error)*100) / (data_obj_1.Units + data_obj_2.Units);
	cout << Success <<" %   success" << endl << endl;
	//cout << WeigthsOut[0][0] << "    " << WeigthsOut[0][1] << "     " << WeigthsOut[0][2]<< endl;               // FOR CHECKING
    }while(Success < 90 && loop <= 20000);									// ITS NOT 20000 FOR PURPOSE OF CHECKING

	if(loop > 20000)
	{
        cout << "Training of network failure !" << endl;
        return false;
	}
	else
		return true;
}

			//  3 HIDDEN LAYERS ALGORITHMS

bool BackPropagationNet::TrainNet_3_hidden( Data & data_obj )      // Training of network
{
    int Error, j, loop = 0, Success;
	cout << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
	cout << "                       TRAINING NETWORK" << endl << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
    do
    {
	    Error = 0;
	    loop ++;
		//Printing the number of loop.
		if( loop < 10 )
            cout <<"Training loop:  " << loop << "       ...   ";
		if( loop >= 10  && loop < 100)
		    cout <<"Training loop:  " << loop << "      ...   ";
		if( loop >= 100 && loop < 1000)
		    cout <<"Training loop:  " << loop << "     ...   ";
		if(	loop >= 1000 && loop < 10000)
			cout <<"Training loop:  " << loop << "    ...   ";
		else if( loop >= 10000 )
			cout << "Training loop:  " << loop << "   ...   ";
        //Train network (do one cycle).
        for(int i = 0; i < data_obj.Units; i++)   // units = 10
        {
	        //Set current input.

	        for(j = 0; j < InputNeurons; j++)
	        {
	            InputLayer[j] = data_obj.Input[i][j];
			}
	        CalculateOutput_3_hidden();
	        int answ_temp = 0, temp = 8;					// This calculate the right outoput and send it to function that compare the network output with the right answer
			for(int k = 0; k < OutputNeurons; k++)
			{
				if(data_obj.Output[i][k] == 1)
				{
					answ_temp += temp;
				}
				temp /= 2;
			}
	        ItIsError(answ_temp);						// this is the comparison
	        //If it was error, change weigths (Error = sum of errors in
			//one cycle of train).
	        if(NetError)
	        {
				Error ++;															// convert integer 0-1 values of data class into float values 0-1
				AdjustWeigths_3_hidden(data_obj.Output[i]);//make an apropriate array sending of func   // WHEN WE ADD DATA FUNCTIONS CHECK THIS FUNCTION THAT IT WILL WORK CORRECT
            }																				// IT USES DATA CLASS FUNCTIONS THAT WE HAVE NOT ADDED YET
	    }
	Success = ((data_obj.Units - Error)*100) / data_obj.Units;
	cout << Success <<" %   success" << endl << endl;
	//cout << WeigthsOut[0][0] << "    " << WeigthsOut[0][1] << "     " << WeigthsOut[0][2]<< endl;               // FOR CHECKING
    }while(Success < 90 && loop <= 20000);									// ITS NOT 20000 FOR PURPOSE OF CHECKING

	if(loop > 20000)
	{
        cout << "Training of network failure !" << endl;
        return false;
	}
	else
		return true;
}

bool BackPropagationNet::TrainRandomNet_3_hidden( Data & data_obj )      // Training of network
{
    int Error, j, loop = 0, Success;
	cout << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
	cout << "                       TRAINING NETWORK" << endl << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
    do
    {
	    Error = 0;
	    loop ++;
		//Printing the number of loop.
		if( loop < 10 )
            cout <<"Training loop:  " << loop << "       ...   ";
		if( loop >= 10  && loop < 100)
		    cout <<"Training loop:  " << loop << "      ...   ";
		if( loop >= 100 && loop < 1000)
		    cout <<"Training loop:  " << loop << "     ...   ";
		if(	loop >= 1000 && loop < 10000)
			cout <<"Training loop:  " << loop << "    ...   ";
		else if( loop >= 10000 )
			cout << "Training loop:  " << loop << "   ...   ";
        //Train network (do one cycle).
        for(int i = 0; i < data_obj.Units; i++)   // units = 10
        {
	        //Set current input.
	        int arr_rand[Numbers];    // ARRAY WITH RANDOM ORDER OF NUMBERS FROM 0 TO 9
			int h = 0;
			while(h < Numbers)      // MAKE THIS ARRAY
			{
				int rand_help = rand()%10;
				bool already_in = false;
				for(int h2 = 0; h2 < h; h2++)
				{
					if(rand_help == arr_rand[h2])
					{
						already_in = true;
					}
				}
				if(!already_in)
				{
					arr_rand[h] = rand_help;
					h++;
				}
			}
	        for(j = 0; j < InputNeurons; j++)
	        {
	            InputLayer[j] = data_obj.Input[arr_rand[i]][j];
			}
	        CalculateOutput_3_hidden();

	        int answ_temp = 0, temp = 8;					// This calculate the right outoput and send it to function that compare the network output with the right answer
			for(int k = 0; k < sizeof(data_obj.Output[arr_rand[i]]); k++)
			{
				if(data_obj.Output[arr_rand[i]][k] == 1)
				{
					answ_temp += temp;
				}
				temp /= 2;
			}
	        ItIsError(answ_temp);						// this is the comparison

	        //If it was error, change weigths (Error = sum of errors in
			//one cycle of train).
	        if(NetError)
	        {
				Error ++;															// convert integer 0-1 values of data class into float values 0-1
				AdjustWeigths_3_hidden(data_obj.Output[arr_rand[i]]);//make an apropriate array sending of func   // WHEN WE ADD DATA FUNCTIONS CHECK THIS FUNCTION THAT IT WILL WORK CORRECT
            }																				// IT USES DATA CLASS FUNCTIONS THAT WE HAVE NOT ADDED YET
	    }

	Success = ((data_obj.Units - Error)*100) / data_obj.Units;
	cout << Success <<" %   success" << endl << endl;

	//cout << WeigthsOut[0][0] << "    " << WeigthsOut[0][1] << "     " << WeigthsOut[0][2]<< endl;               // FOR CHECKING
    }while(Success < 90 && loop <= 20000);									// ITS NOT 20000 FOR PURPOSE OF CHECKING

	if(loop > 20000)
	{
        cout << "Training of network failure !" << endl;
        return false;
	}
	else
		return true;
}

bool BackPropagationNet::TrainNet_2_groups_3_hidden( Data & data_obj_1, Data & data_obj_2)      // Training of network
{
    int Error, j, loop = 0, Success;
	cout << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
	cout << "                       TRAINING NETWORK" << endl << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
    do
    {
	    Error = 0;
	    loop ++;
		//Printing the number of loop.
		if( loop < 10 )
            cout <<"Training loop:  " << loop << "       ...   ";
		if( loop >= 10  && loop < 100)
		    cout <<"Training loop:  " << loop << "      ...   ";
		if( loop >= 100 && loop < 1000)
		    cout <<"Training loop:  " << loop << "     ...   ";
		if(	loop >= 1000 && loop < 10000)
			cout <<"Training loop:  " << loop << "    ...   ";
		else if( loop >= 10000 )
			cout << "Training loop:  " << loop << "   ...   ";
        //Train network (do one cycle).
        for(int i = 0; i < data_obj_1.Units; i++)   // units = 10
        {
	        //Set current input.

	        for(j = 0; j < InputNeurons; j++)
	        {
	            InputLayer[j] = data_obj_1.Input[i][j];
			}
	        CalculateOutput_3_hidden();
	        int answ_temp = 0, temp = 8;					// This calculate the right outoput and send it to function that compare the network output with the right answer
			for(int k = 0; k < OutputNeurons; k++)
			{
				if(data_obj_1.Output[i][k] == 1)
				{
					answ_temp += temp;
				}
				temp /= 2;
			}
	        ItIsError(answ_temp);						// this is the comparison
	        //If it was error, change weigths (Error = sum of errors in
			//one cycle of train).
	        if(NetError)
	        {
				Error ++;															// convert integer 0-1 values of data class into float values 0-1
				AdjustWeigths_3_hidden(data_obj_1.Output[i]);//make an apropriate array sending of func   // WHEN WE ADD DATA FUNCTIONS CHECK THIS FUNCTION THAT IT WILL WORK CORRECT
            }																				// IT USES DATA CLASS FUNCTIONS THAT WE HAVE NOT ADDED YET
	    }
	    for(int i = 0; i < data_obj_2.Units; i++)   // units = 10
        {
	        //Set current input.

	        for(j = 0; j < InputNeurons; j++)
	        {
	            InputLayer[j] = data_obj_2.Input[i][j];
			}
	        CalculateOutput_3_hidden();
	        int answ_temp = 0, temp = 8;					// This calculate the right outoput and send it to function that compare the network output with the right answer
			for(int k = 0; k < OutputNeurons; k++)
			{
				if(data_obj_2.Output[i][k] == 1)
				{
					answ_temp += temp;
				}
				temp /= 2;
			}
	        ItIsError(answ_temp);						// this is the comparison
	        //If it was error, change weigths (Error = sum of errors in
			//one cycle of train).
	        if(NetError)
	        {
				Error ++;															// convert integer 0-1 values of data class into float values 0-1
				AdjustWeigths_3_hidden(data_obj_2.Output[i]);//make an apropriate array sending of func   // WHEN WE ADD DATA FUNCTIONS CHECK THIS FUNCTION THAT IT WILL WORK CORRECT
            }																				// IT USES DATA CLASS FUNCTIONS THAT WE HAVE NOT ADDED YET
	    }
	Success = ((data_obj_1.Units + data_obj_2.Units - Error)*100) / (data_obj_1.Units + data_obj_2.Units);
	cout << Success <<" %   success" << endl << endl;
	//cout << WeigthsOut[0][0] << "    " << WeigthsOut[0][1] << "     " << WeigthsOut[0][2]<< endl;               // FOR CHECKING
    }while(Success < 90 && loop <= 20000);									// ITS NOT 20000 FOR PURPOSE OF CHECKING

	if(loop > 20000)
	{
        cout << "Training of network failure !" << endl;
        return false;
	}
	else
		return true;
}

bool BackPropagationNet::TrainRandomNet_2_groups_3_hidden( Data & data_obj_1, Data & data_obj_2)      // Training of network
{
    int Error, j, loop = 0, Success;
	cout << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
	cout << "                       TRAINING NETWORK" << endl << endl;
	cout << "   --------------------------------------------------------";
	cout << endl << endl;
    do
    {
	    Error = 0;
	    loop ++;
		//Printing the number of loop.
		if( loop < 10 )
            cout <<"Training loop:  " << loop << "       ...   ";
		if( loop >= 10  && loop < 100)
		    cout <<"Training loop:  " << loop << "      ...   ";
		if( loop >= 100 && loop < 1000)
		    cout <<"Training loop:  " << loop << "     ...   ";
		if(	loop >= 1000 && loop < 10000)
			cout <<"Training loop:  " << loop << "    ...   ";
		else if( loop >= 10000 )
			cout << "Training loop:  " << loop << "   ...   ";
        //Train network (do one cycle).
        for(int i = 0; i < data_obj_1.Units; i++)   // units = 10
        {
	        //Set current input.
	        int arr_rand[Numbers];    // ARRAY WITH RANDOM ORDER OF NUMBERS FROM 0 TO 9
			int h = 0;
			while(h < Numbers)      // MAKE THIS ARRAY
			{
				int rand_help = rand()%10;
				bool already_in = false;
				for(int h2 = 0; h2 < h; h2++)
				{
					if(rand_help == arr_rand[h2])
					{
						already_in = true;
					}
				}
				if(!already_in)
				{
					arr_rand[h] = rand_help;
					h++;
				}
			}
	        for(j = 0; j < InputNeurons; j++)
	        {
	            InputLayer[j] = data_obj_1.Input[arr_rand[i]][j];
			}
	        CalculateOutput_3_hidden();
	        int answ_temp = 0, temp = 8;					// This calculate the right outoput and send it to function that compare the network output with the right answer
			for(int k = 0; k < OutputNeurons; k++)
			{
				if(data_obj_1.Output[arr_rand[i]][k] == 1)
				{
					answ_temp += temp;
				}
				temp /= 2;
			}
	        ItIsError(answ_temp);						// this is the comparison
	        //If it was error, change weigths (Error = sum of errors in
			//one cycle of train).
	        if(NetError)
	        {
				Error ++;															// convert integer 0-1 values of data class into float values 0-1
				AdjustWeigths_3_hidden(data_obj_1.Output[arr_rand[i]]);//make an apropriate array sending of func   // WHEN WE ADD DATA FUNCTIONS CHECK THIS FUNCTION THAT IT WILL WORK CORRECT
            }																				// IT USES DATA CLASS FUNCTIONS THAT WE HAVE NOT ADDED YET
	    }

	    for(int i = 0; i < data_obj_2.Units; i++)   // units = 10
        {
	        //Set current input for second group.
	        int arr_rand[Numbers];    // ARRAY WITH RANDOM ORDER OF NUMBERS FROM 0 TO 9
			int h = 0;
	        while(h < Numbers)      // MAKE THIS ARRAY RANDOM FOR SECOND GROUP
			{
				int rand_help = rand()%10;
				bool already_in = false;
				for(int h2 = 0; h2 < h; h2++)
				{
					if(rand_help == arr_rand[h2])
					{
						already_in = true;
					}
				}
				if(!already_in)
				{
					arr_rand[h] = rand_help;
					h++;
				}
			}
	        for(j = 0; j < InputNeurons; j++)
	        {
	            InputLayer[j] = data_obj_2.Input[arr_rand[i]][j];
			}
	        CalculateOutput_3_hidden();
	        int answ_temp = 0, temp = 8;					// This calculate the right outoput and send it to function that compare the network output with the right answer
			for(int k = 0; k < OutputNeurons; k++)
			{
				if(data_obj_2.Output[arr_rand[i]][k] == 1)
				{
					answ_temp += temp;
				}
				temp /= 2;
			}
	        ItIsError(answ_temp);						// this is the comparison
	        //If it was error, change weigths (Error = sum of errors in
			//one cycle of train).
	        if(NetError)
	        {
				Error ++;															// convert integer 0-1 values of data class into float values 0-1
				AdjustWeigths_3_hidden(data_obj_2.Output[arr_rand[i]]);//make an apropriate array sending of func   // WHEN WE ADD DATA FUNCTIONS CHECK THIS FUNCTION THAT IT WILL WORK CORRECT
            }																				// IT USES DATA CLASS FUNCTIONS THAT WE HAVE NOT ADDED YET
	    }
	Success = ((data_obj_1.Units + data_obj_2.Units - Error)*100) / (data_obj_1.Units + data_obj_2.Units);
	cout << Success <<" %   success" << endl << endl;
	//cout << WeigthsOut[0][0] << "    " << WeigthsOut[0][1] << "     " << WeigthsOut[0][2]<< endl;               // FOR CHECKING
    }while(Success < 90 && loop <= 20000);									// ITS NOT 20000 FOR PURPOSE OF CHECKING

	if(loop > 20000)
	{
        cout << "Training of network failure !" << endl;
        return false;
	}
	else
		return true;
}

int BackPropagationNet::TestNet(Data & data_obj)     // Testing network one time
{  													//
    int Error = 0, j, Success;

	cout << endl << endl << endl;
	cout << "---------------------------------------------------------------------";
	cout << endl << endl;
    cout << "                    TEST NETWORK" << endl << endl;
	cout << "---------------------------------------------------------------------";
	cout << endl << endl;
    cout << "Test network    ...  " << endl;
    //Train network (do one cycle).
    for(int i = 0; i < data_obj.Units; i++) // transform number in form of matrix into array of neurons
    {
        //Set current input.
	    for(j = 0; j < InputNeurons; j++)
	    {
	        InputLayer[j] = data_obj.Input[i][j];
	    }
	    CalculateOutput(); // CALCULATE OUTLAYER
	    int answ_temp = 0, temp = 8;					// This calculate the right outoput and send it to function that compare the network output with the right answer
		for(int k = 0; k < OutputNeurons; k++)
		{
			if(data_obj.Output[i][k] == 1)
			{
				answ_temp += temp;
			}
			temp /= 2;
		}
		ItIsError(answ_temp);// check error
		//Error = sum of errors in this one cycle of test.
	    if(NetError)
	    {
	      Error ++;
		}
    }
    Success = ((data_obj.Units - Error)*100) / data_obj.Units;
    cout << Success << "%   success" << endl;
    return Success;
}

int BackPropagationNet::TestNet_3_hidden(Data & data_obj)     // Testing network one time
{  													//
    int Error = 0, j, Success;

	cout << endl << endl << endl;
	cout << "---------------------------------------------------------------------";
	cout << endl << endl;
    cout << "                    TEST NETWORK" << endl << endl;
	cout << "---------------------------------------------------------------------";
	cout << endl << endl;
    cout << "Test network    ...  " << endl;
    //Train network (do one cycle).
    for(int i = 0; i < data_obj.Units; i++) // transform number in form of matrix into array of neurons
    {
        //Set current input.
	    for(j = 0; j < InputNeurons; j++)
	    {
	        InputLayer[j] = data_obj.Input[i][j];
	    }
	    CalculateOutput_3_hidden(); // CALCULATE OUTLAYER
	    int answ_temp = 0, temp = 8;					// This calculate the right outoput and send it to function that compare the network output with the right answer
		for(int k = 0; k < OutputNeurons; k++)
		{
			if(data_obj.Output[i][k] == 1)
			{
				answ_temp += temp;
			}
			temp /= 2;
		}
		ItIsError(answ_temp);// check error
		//Error = sum of errors in this one cycle of test.
	    if(NetError)
	    {
	      Error ++;
		}
    }
    Success = ((data_obj.Units - Error)*100) / data_obj.Units;
    cout << Success << "%   success" << endl;
    return Success;
}

//************************ CLASS DATA *************************************

Data::Data ()
{
    Units = 0;
}

//_________________________________________________________________________

Data::~Data()
{
    Reset ();
}

//_________________________________________________________________________

void Data::Reset()
{
    Units = 0;
    delete[] Input;
    delete[] Output;
}

//_________________________________________________________________________

bool Data::SetInputOutput(char In[][y][x], char Out[][5], int num_patterns)
{
	int n, i, j;
    if ( Units != num_patterns)
	{
		 if (Units)
		 {
			Reset();
		 }
		 /*for(int i = 0; i < num_patterns; i++)
		 {
			Input[i] = new int[InputNeurons];
			cout << i << "WE are on this raw" << endl;
		 }*/
         /*if (!(Input = new int[num_patterns]))                                 // PROBLEM IS HERE
         {
             cout << "Insufficient memory for Input" << endl;
             return false;
         }*/
         /*if (!(Output = new OutArr[OutputNeurons]))
         {
             cout << "Insufficient memory for Output" << endl;
			 delete [] Input;
             return false;
         }*/
		 Units = num_patterns;
	}

    for(n = 0; n < Units; n++)                         //Set input vectors.
    {
	    for(i = 0; i < y; i++)
        {
	        for(j = 0; j < (x - 1); j++)
		    {
		        Input[n][i * (x - 1) + j] = (In[n][i][j] == '*') ? Hi : Low; // THIS IS CHECKING ON INPUT ARRAY AND PUT A HI IF ITS * SYMBOL
			}
	    }
    }

	//Set corresponding to input expected output.
    for(i = 0; i < Units; i++)
    {
		for(j = 0; j < DataOutputs; j++)
		{
	        Output[i][j] = (Out[i][j] == '*') ? Hi : Low;
		}
    }

    return true;
}



//    MAIN		MAIN		MAIN		MAIN		MAIN		MAIN		MAIN		MAIN		MAIN

int main()
{
	Data data_obj, data_obj_2, data_check;
    BackPropagationNet back_prop_obj;
	cout << "Back Propagation Network" << endl << endl;
	cout << "Currently all of the programs are commented" << endl;
	cout << "Please head to main algorithm and chose what type of program you want to run" << endl;
	cout << "You have options of 1 or 3 hidden layers" << endl;
	cout << "Random or serial order of training" << endl;
	cout << "And 3 parts of the problem given in course final project" << endl;

	     //  THIS IS LEARNING GROUP - REGULAR NUMBERS, TEST GROUP - ERROR NUMBERS.
	   // task 1 part 1 - serial numbers order
	/*
	if(! data_obj.SetInputOutput(InputPattern, OutputPattern, Numbers))
		return 0;
	back_prop_obj.Initialize();
	bool flag = back_prop_obj.TrainNet(data_obj);
	if(flag)
	{
		cout << "Studied network complete" << endl;
		// now we check the other groups with our network
		if(! data_obj.SetInputOutput(InputErrPattern1, OutputPattern, Numbers))  // CHECK FIRST GROUP WITH ERROR
			return 0;
		int suc1 = back_prop_obj.TestNet(data_obj);
		cout << suc1 << " is the success on the first error group" << endl;

		if(! data_obj.SetInputOutput(InputErrPattern2, OutputPattern, Numbers)) // CHECK SECOND GROUP WITH ERROR
			return 0;
		int suc2 = back_prop_obj.TestNet(data_obj);
		cout << suc2 << " is the success on the first error group" << endl;

		if(! data_obj.SetInputOutput(InputErrPattern3, OutputPattern, Numbers))  // CHECK THIRD GROUP WITH ERROR
			return 0;
		int suc3 = back_prop_obj.TestNet(data_obj);
		cout << suc3 << " is the success on the first error group" << endl;
		//THIS IS THE FISRT TASK, SERIAL ORDER
	}
	else
	{
		cout << "Studing not good, please try again later";
	}
	*/



	 //  THIS IS LEARNING GROUP - REGULAR NUMBERS, TEST GROUP - ERROR NUMBERS.
	   // task 1 part 2 - random numbers order
	/*
	   if(! data_obj.SetInputOutput(InputPattern, OutputPattern, Numbers))
		return 0;
	back_prop_obj.Initialize();
	bool flag = back_prop_obj.TrainRandomNet(data_obj);
	if(flag)
	{
		cout << "Studied network complete" << endl;
		// now we check the other groups with our network
		if(! data_obj.SetInputOutput(InputErrPattern1, OutputPattern, Numbers))  // CHECK FIRST GROUP WITH ERROR
			return 0;
		int suc1 = back_prop_obj.TestNet(data_obj);
		cout << suc1 << " is the success on the first error group" << endl;

		if(! data_obj.SetInputOutput(InputErrPattern2, OutputPattern, Numbers)) // CHECK SECOND GROUP WITH ERROR
			return 0;
		int suc2 = back_prop_obj.TestNet(data_obj);
		cout << suc2 << " is the success on the first error group" << endl;

		if(! data_obj.SetInputOutput(InputErrPattern3, OutputPattern, Numbers))  // CHECK THIRD GROUP WITH ERROR
			return 0;
		int suc3 = back_prop_obj.TestNet(data_obj);
		cout << suc3 << " is the success on the first error group" << endl;
		//THIS IS THE FISRT TASK, RANDOM ORDER
	}
	else
	{
		cout << "Studing not good, please try again later";
	}
	 // this is end of task 1 part 2
	*/


		// THIS IS LEARNING GROUP - REGULAR NUMBERS AND ERROR NUMBERS
		// Task 2, part 1 - serial number order
	/*
	if(! data_obj.SetInputOutput(InputPattern, OutputPattern, Numbers))
	{
		return 0;
	}
	if(! data_obj_2.SetInputOutput(InputErrPattern1, OutputPattern, Numbers))
	{
		return 0;
	}
	back_prop_obj.Initialize();
	bool flag = back_prop_obj.TrainNet_2_groups(data_obj, data_obj_2);
	if(flag)
	{
		cout << "Studied network complete" << endl;
		// now we check the other groups with our network
		if(!data_check.SetInputOutput(InputErrPattern2, OutputPattern, Numbers)) // CHECK SECOND GROUP WITH ERROR
		{
			cout << "Inside of main class if that check data_obj set input" << endl;
			return 0;
		}
		int suc2 = back_prop_obj.TestNet(data_check);
		cout << suc2 << " is the success on the first error group" << endl;

		if(!data_obj.SetInputOutput(InputErrPattern3, OutputPattern, Numbers))  // CHECK THIRD GROUP WITH ERROR
			return 0;
		int suc3 = back_prop_obj.TestNet(data_obj);
		cout << suc3 << " is the success on the first error group" << endl;
		//THIS IS THE SECOND TASK, SERIAL ORDER.
	}
	else
	{
		cout << "Studing not good, please try again later";
	}
	*/

		// THIS IS LEARNING GROUP - REGULAR NUMBERS AND ERROR NUMBERS
		// Task 2, part 2 - random number order
	/*
	if(! data_obj.SetInputOutput(InputPattern, OutputPattern, Numbers))
	{
		return 0;
	}
	if(! data_obj_2.SetInputOutput(InputErrPattern1, OutputPattern, Numbers))
	{
		return 0;
	}
	back_prop_obj.Initialize();
	bool flag = back_prop_obj.TrainRandomNet_2_groups(data_obj, data_obj_2);
	if(flag)
	{
		cout << "Studied network complete" << endl;
		// now we check the other groups with our network
		if(!data_check.SetInputOutput(InputErrPattern2, OutputPattern, Numbers)) // CHECK SECOND GROUP WITH ERROR
		{
			cout << "Inside of main class if that check data_obj set input" << endl;
			return 0;
		}
		int suc2 = back_prop_obj.TestNet(data_check);
		cout << suc2 << " is the success on the second error group" << endl;

		if(!data_obj.SetInputOutput(InputErrPattern3, OutputPattern, Numbers))  // CHECK THIRD GROUP WITH ERROR
			return 0;
		int suc3 = back_prop_obj.TestNet(data_obj);
		cout << suc3 << " is the success on the third error group" << endl;
		//THIS IS THE SECOND TASK, SERIAL ORDER.
	}
	else
	{
		cout << "Studing not good, please try again later";
	}
	*/


		// THIS IS LEARNING GROUP - ERROR NUMBERS AND ERROR NUMBERS
		// Task 3, part 1 - serial number order
	/*
	if(! data_obj.SetInputOutput(InputErrPattern1, OutputPattern, Numbers))
	{
		return 0;
	}
	if(! data_obj_2.SetInputOutput(InputErrPattern2, OutputPattern, Numbers))
	{
		return 0;
	}
	back_prop_obj.Initialize();
	bool flag = back_prop_obj.TrainNet_2_groups(data_obj, data_obj_2);
	if(flag)
	{
		cout << "Studied network complete" << endl;
		// now we check the other groups with our network

		if(!data_obj.SetInputOutput(InputErrPattern3, OutputPattern, Numbers))  // CHECK THIRD GROUP WITH ERROR
			return 0;
		int suc3 = back_prop_obj.TestNet(data_obj);
		cout << suc3 << " is the success on the third error group" << endl;
		//THIS IS THE SECOND TASK, SERIAL ORDER.
	}
	else
	{
		cout << "Studing not good, please try again later";
	}
	*/

		// THIS IS LEARNING GROUP - ERROR NUMBERS AND ERROR NUMBERS
		// Task 3, part 2 - random number order
	/*
	if(!data_obj.SetInputOutput(InputErrPattern1, OutputPattern, Numbers))
	{
		return 0;
	}
	if(!data_obj_2.SetInputOutput(InputErrPattern2, OutputPattern, Numbers))
	{
		return 0;
	}
	back_prop_obj.Initialize();
	bool flag = back_prop_obj.TrainRandomNet_2_groups(data_obj, data_obj_2);
	if(flag)
	{
		cout << "Studied network complete" << endl;
		// now we check the other groups with our network

		if(!data_obj.SetInputOutput(InputErrPattern3, OutputPattern, Numbers))  // CHECK THIRD GROUP WITH ERROR
			return 0;
		int suc3 = back_prop_obj.TestNet(data_obj);
		cout << suc3 << " is the success on the third error group" << endl;
		//THIS IS THE SECOND TASK, SERIAL ORDER.
	}
	else
	{
		cout << "Studing not good, please try again later";
	}
	*/

				//     3 HIDDEN LAYERS
		//  THIS IS LEARNING GROUP - REGULAR NUMBERS, TEST GROUP - ERROR NUMBERS. 3 HIDDEN LAYERS
	   // task 1 part 1 - serial numbers order, 3 hidden layers
	/*
	if(! data_obj.SetInputOutput(InputPattern, OutputPattern, Numbers))
		return 0;
	back_prop_obj.Initialize();
	cout << "HELOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO" << endl << endl;
	bool flag = back_prop_obj.TrainNet_3_hidden(data_obj);
	if(flag)
	{
		cout << "Studied network complete" << endl;
		// now we check the other groups with our network
		if(! data_obj.SetInputOutput(InputErrPattern1, OutputPattern, Numbers))  // CHECK FIRST GROUP WITH ERROR
			return 0;
		int suc1 = back_prop_obj.TestNet_3_hidden(data_obj);
		cout << suc1 << " is the success on the first error group" << endl;

		if(! data_obj.SetInputOutput(InputErrPattern2, OutputPattern, Numbers)) // CHECK SECOND GROUP WITH ERROR
			return 0;
		int suc2 = back_prop_obj.TestNet_3_hidden(data_obj);
		cout << suc2 << " is the success on the first error group" << endl;

		if(! data_obj.SetInputOutput(InputErrPattern3, OutputPattern, Numbers))  // CHECK THIRD GROUP WITH ERROR
			return 0;
		int suc3 = back_prop_obj.TestNet_3_hidden(data_obj);
		cout << suc3 << " is the success on the first error group" << endl;
		//THIS IS THE FISRT TASK, SERIAL ORDER
	}
	else
	{
		cout << "Studing not good, please try again later";
	}
	*/

		 //  THIS IS LEARNING GROUP - REGULAR NUMBERS, TEST GROUP - ERROR NUMBERS. 3 HIDDEN LAYERS
	   // task 1 part 2 - random numbers order, 3 hidden layers
	/*
	   if(! data_obj.SetInputOutput(InputPattern, OutputPattern, Numbers))
		return 0;
	back_prop_obj.Initialize();
	bool flag = back_prop_obj.TrainRandomNet_3_hidden(data_obj);
	if(flag)
	{
		cout << "Studied network complete" << endl;
		// now we check the other groups with our network
		if(! data_obj.SetInputOutput(InputErrPattern1, OutputPattern, Numbers))  // CHECK FIRST GROUP WITH ERROR
			return 0;
		int suc1 = back_prop_obj.TestNet_3_hidden(data_obj);
		cout << suc1 << " is the success on the first error group" << endl;

		if(! data_obj.SetInputOutput(InputErrPattern2, OutputPattern, Numbers)) // CHECK SECOND GROUP WITH ERROR
			return 0;
		int suc2 = back_prop_obj.TestNet_3_hidden(data_obj);
		cout << suc2 << " is the success on the first error group" << endl;

		if(! data_obj.SetInputOutput(InputErrPattern3, OutputPattern, Numbers))  // CHECK THIRD GROUP WITH ERROR
			return 0;
		int suc3 = back_prop_obj.TestNet_3_hidden(data_obj);
		cout << suc3 << " is the success on the first error group" << endl;
		//THIS IS THE FISRT TASK, RANDOM ORDER
	}
	else
	{
		cout << "Studing not good, please try again later";
	}
	 // this is end of task 1 part 2
	*/

		// THIS IS LEARNING GROUP - REGULAR NUMBERS AND ERROR NUMBERS. 3 HIDDEN LAYER
		// Task 2, part 1 - serial number order, 3 hidden layer
	/*
	if(! data_obj.SetInputOutput(InputPattern, OutputPattern, Numbers))
	{
		return 0;
	}
	if(! data_obj_2.SetInputOutput(InputErrPattern1, OutputPattern, Numbers))
	{
		return 0;
	}
	back_prop_obj.Initialize();
	bool flag = back_prop_obj.TrainNet_2_groups_3_hidden(data_obj, data_obj_2);
	if(flag)
	{
		cout << "Studied network complete" << endl;
		// now we check the other groups with our network
		if(!data_check.SetInputOutput(InputErrPattern2, OutputPattern, Numbers)) // CHECK SECOND GROUP WITH ERROR
		{
			cout << "Inside of main class if that check data_obj set input" << endl;
			return 0;
		}
		int suc2 = back_prop_obj.TestNet_3_hidden(data_check);
		cout << suc2 << " is the success on the first error group" << endl;

		if(!data_obj.SetInputOutput(InputErrPattern3, OutputPattern, Numbers))  // CHECK THIRD GROUP WITH ERROR
			return 0;
		int suc3 = back_prop_obj.TestNet_3_hidden(data_obj);
		cout << suc3 << " is the success on the first error group" << endl;
		//THIS IS THE SECOND TASK, SERIAL ORDER.
	}
	else
	{
		cout << "Studing not good, please try again later";
	}
	/*

		// THIS IS LEARNING GROUP - REGULAR NUMBERS AND ERROR NUMBERS. 3 HIDDEN LAYERS
		// Task 2, part 2 - random number order, 3 hidden layers
	/*
	if(! data_obj.SetInputOutput(InputPattern, OutputPattern, Numbers))
	{
		return 0;
	}
	if(! data_obj_2.SetInputOutput(InputErrPattern1, OutputPattern, Numbers))
	{
		return 0;
	}
	back_prop_obj.Initialize();
	bool flag = back_prop_obj.TrainRandomNet_2_groups_3_hidden(data_obj, data_obj_2);
	if(flag)
	{
		cout << "Studied network complete" << endl;
		// now we check the other groups with our network
		if(!data_check.SetInputOutput(InputErrPattern2, OutputPattern, Numbers)) // CHECK SECOND GROUP WITH ERROR
		{
			cout << "Inside of main class if that check data_obj set input" << endl;
			return 0;
		}
		int suc2 = back_prop_obj.TestNet_3_hidden(data_check);
		cout << suc2 << " is the success on the second error group" << endl;

		if(!data_obj.SetInputOutput(InputErrPattern3, OutputPattern, Numbers))  // CHECK THIRD GROUP WITH ERROR
			return 0;
		int suc3 = back_prop_obj.TestNet_3_hidden(data_obj);
		cout << suc3 << " is the success on the third error group" << endl;
		//THIS IS THE SECOND TASK, SERIAL ORDER.
	}
	else
	{
		cout << "Studing not good, please try again later";
	}
	*/

			// THIS IS LEARNING GROUP - REGULAR NUMBERS AND ERROR NUMBERS. 3 HIDDEN LAYER
		// Task 2, part 2 - random number order, 3 hidden layer
	/*
	if(! data_obj.SetInputOutput(InputPattern, OutputPattern, Numbers))
	{
		return 0;
	}
	if(! data_obj_2.SetInputOutput(InputErrPattern1, OutputPattern, Numbers))
	{
		return 0;
	}
	back_prop_obj.Initialize();
	bool flag = back_prop_obj.TrainRandomNet_2_groups_3_hidden(data_obj, data_obj_2);
	if(flag)
	{
		cout << "Studied network complete" << endl;
		// now we check the other groups with our network
		if(!data_check.SetInputOutput(InputErrPattern2, OutputPattern, Numbers)) // CHECK SECOND GROUP WITH ERROR
		{
			cout << "Inside of main class if that check data_obj set input" << endl;
			return 0;
		}
		int suc2 = back_prop_obj.TestNet_3_hidden(data_check);
		cout << suc2 << " is the success on the second error group" << endl;

		if(!data_obj.SetInputOutput(InputErrPattern3, OutputPattern, Numbers))  // CHECK THIRD GROUP WITH ERROR
			return 0;
		int suc3 = back_prop_obj.TestNet_3_hidden(data_obj);
		cout << suc3 << " is the success on the third error group" << endl;
		//THIS IS THE SECOND TASK, SERIAL ORDER.
	}
	else
	{
		cout << "Studing not good, please try again later";
	}
	*/

		// THIS IS LEARNING GROUP - ERROR NUMBERS AND ERROR NUMBERS. 3 HIDDEN LAYER
		// Task 3, part 1 - serial number order, 3 hidden layer
	/*
	if(! data_obj.SetInputOutput(InputErrPattern1, OutputPattern, Numbers))
	{
		return 0;
	}
	if(! data_obj_2.SetInputOutput(InputErrPattern2, OutputPattern, Numbers))
	{
		return 0;
	}
	back_prop_obj.Initialize();
	bool flag = back_prop_obj.TrainNet_2_groups_3_hidden(data_obj, data_obj_2);
	if(flag)
	{
		cout << "Studied network complete" << endl;
		// now we check the other groups with our network

		if(!data_obj.SetInputOutput(InputErrPattern3, OutputPattern, Numbers))  // CHECK THIRD GROUP WITH ERROR
			return 0;
		int suc3 = back_prop_obj.TestNet_3_hidden(data_obj);
		cout << suc3 << " is the success on the third error group" << endl;
		//THIS IS THE SECOND TASK, SERIAL ORDER.
	}
	else
	{
		cout << "Studing not good, please try again later";
	}
	*/

		// THIS IS LEARNING GROUP - ERROR NUMBERS AND ERROR NUMBERS. 3 HIDDEN LAYER
		// Task 3, part 2 - random number order, 3 hidden layer

	if(!data_obj.SetInputOutput(InputErrPattern1, OutputPattern, Numbers))
	{
		return 0;
	}
	if(!data_obj_2.SetInputOutput(InputErrPattern2, OutputPattern, Numbers))
	{
		return 0;
	}
	back_prop_obj.Initialize();
	bool flag = back_prop_obj.TrainRandomNet_2_groups_3_hidden(data_obj, data_obj_2);
	if(flag)
	{
		cout << "Studied network complete" << endl;
		if(!data_obj.SetInputOutput(InputErrPattern3, OutputPattern, Numbers))  // CHECK THIRD GROUP WITH ERROR
			return 0;
		int suc3 = back_prop_obj.TestNet_3_hidden(data_obj);
		cout << suc3 << " is the success on the third error group" << endl;
	}
	else
	{
		cout << "Studing not good, please try again later";
	}

	return 0;
}

