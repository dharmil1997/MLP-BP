
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include "time.h"
using namespace std;

int trainingData[12][4][3] = {0};
int testingData[14][4][3] = {0};
int targetData[12][4] = {0};
int resultData[14][4] = {0};
int epoch = 0;
double net[3] = {0};
double ao[3] = {0};
double delta[3] = {0};
double weights[3][3] = {0};
ifstream inputs;


void input_training_data();
void input_target_data();
void input_testing_data();
void generate_weights();
void reset_variables();
float activation_function(float x);
void neural_net_training();
void hidden_layer_calc(int calc);
void outer_layer_calc();
void err_calc_output_layer();
void err_calc_hidden_layer();
void update_weights();
float avg_err_output_layer();


int main()
{
    generate_weights();
    neural_net_training();
    return 0;
}


void input_training_data()
{
    inputs.open("train-2pl.txt");
    while(!inputs.eof())
    {
        for(auto i = 0; i < 12; i++)
        {
            for(auto j = 0; j < 4; j++)
            {
                for(auto k = 1; k < 3; k++)
                {
                    trainingData[i][j][0] = 1;
                    inputs >> trainingData[i][j][k];
                }
            }
        }
    }
    inputs.close();
}


void input_target_data()
{
    inputs.open("target-2pl.txt");
    
    while(!inputs.eof())
    {
        for(auto i = 0; i < 12; i++)
        {
            for (auto j = 0; j < 4; j++) {
                inputs >> targetData[i][j];
            }
        }
    }
    inputs.close();
}


void input_testing_data()
{
    inputs.open("test-2pl.txt");
    while(!inputs.eof())
    {
        for(auto i = 0; i < 14; i++)
        {
            for(auto j = 0; j < 4; j++)
            {
                for(auto k = 1; k < 3; k++)
                {
                    testingData[i][j][0] = 1;
                    inputs >> testingData[i][j][k];
                }
            }
        }
    }
    inputs.close();
}

void generate_weights()
{
    srand(static_cast<unsigned int>(time(NULL)));
    
    for(auto i = 0; i < 3; i++)
    {
        for(auto j = 0; j < 3; j++)
        {
            int randNum = rand() % 2;
            if (randNum == 1)
                weights[i][j] = -1 * (double(rand()) / (double(RAND_MAX) + 1.0));
            else
                weights[i][j] = double(rand()) / (double(RAND_MAX) + 1.0);
        }
    }
}


void reset_variables()
{
    for(int i = 0; i < 3; i++)
    {
        ao[i] = 0;
        net[i] = 0;
        delta[i] = 0;
    }
}


void neural_net_training()
{
    while(true)
    {
        input_training_data();
        input_target_data();
        
        reset_variables();
        
        for(auto i = 0; i < 12; i++)
        {
            for(auto j = 0; j < 4; j++)
            {
                for(auto k = 0; k < 3; k++)
                {
                    hidden_layer_calc(trainingData[i][j][k]);
                    outer_layer_calc();
                }
            }
        }
        err_calc_output_layer();
        err_calc_hidden_layer();
        update_weights();
        
        float avg_err = avg_err_output_layer();
        if(avg_err < 0)
        {
            avg_err = avg_err * -1;
        }
        reset_variables();
        epoch++;
        
        if(avg_err < 1 && epoch >= 1000)
            break;
    }
    
    input_testing_data();
    reset_variables();
    
    for(auto i = 0; i < 14; i++)
    {
        for(auto j = 0; j < 4; j++)
        {
            for(auto k = 0; k < 3; k++)
            {
                hidden_layer_calc(testingData[i][j][k]);
                outer_layer_calc();
            }
            if(ao[2] < 0.5)
                resultData[i][j] = 0;
            else
                resultData[i][j] = 1;
        }
    }
    for(auto i = 0; i < 12; i++)
    {
        for (auto j = 0; j < 4; j++)
        {
            cout << resultData[i][j];
        }
    }
}


float activation_function(float x)
{
    float sigmoid = 0.5 * tanh(x) + 0.5;
    return sigmoid;
}


void hidden_layer_calc(int calc)
{
    net[0] = weights[2][0];
    for(auto i = 0; i < 2; i++)
    {
        for(auto j = 0; j < 3; j++)
        {
            net[i] = net[i] + weights[i][j] * calc;
        }
        ao[i] = activation_function(net[i]);
    }
}


void outer_layer_calc()
{
    net[0] = weights[2][0];
    for(auto i = 0; i < 3; i++)
    {
        if(i == 0)
        {
            net[2] = weights[2][0] * weights[2][0];
        }
        else
        {
            net[2] = net[2] + weights[2][i] * ao[i-1];
            if(i == 2)
            {
                ao[2] = activation_function(net[2]);
            }
        }
    }
}



void err_calc_hidden_layer()
{
    float deltaSum = 0.0;
    deltaSum = deltaSum + delta[2] * weights[2][2];
    for(auto k = 0; k < 2; k ++)
    {
        delta[k] = ao[k] * (1.0-ao[k]) * deltaSum;
    }
}


void err_calc_output_layer()
{
    for(auto i = 0; i < 12; i++)
    {
        for(auto j = 0; j < 4; j++)
        {
            delta[2] = ao[2] * (1.0 - ao[2]) * (targetData[i][j] - ao[2]);
        }
    }
}


void update_weights()
{
    srand(static_cast<unsigned int>(time(NULL)));
    int gain = 0;
    
    for(auto i = 0; i < 3; i++)
    {
        gain = double(rand()) / (double(RAND_MAX) + 1.0);
        if (i == 0)
            weights[2][0] = weights[2][0] + gain * delta[2] * weights[2][0];
        else
            weights[2][i] = weights[2][i] + gain * delta[2] * ao[i-1];
    }
    
    gain = 0;
    
    for(auto j = 0; j < 2; j++)
    {
        for(auto k = 0; k < 3; k++)
        {
            for(auto x = 0; x < 12; x++)
            {
                for(auto y = 0; y < 4; y++)
                {
                    for(auto z = 0; z < 3; z++)
                    {
                        gain = double(rand()) / (double(RAND_MAX) + 1.0);
                        weights[j][k] = weights[j][k] + gain * delta[j] * trainingData[x][y][z];
                    }
                }
            }
        }
    }
}


float avg_err_output_layer()
{
    float avgErr = 0;
    
    for(auto i = 0; i < 12; i++)
    {
        for(auto j = 0; j < 4; j++)
        {
            avgErr = (avgErr + ao[2] * (1.0 - ao[2]) * (targetData[i][j] - ao[2]));
        }
    }

    return avgErr/4;
}
