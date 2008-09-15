#include <gtest/gtest.h>
#include <iostream>

#include "arac.h"

namespace AracTesting {


TEST(TestBuildNetwork, IdentityConnectionEquality) {
    Layer* inlayer_p = make_layer(2, 2);
    Layer* outlayer_p = make_layer(2, 2);
    
    identity_connect(inlayer_p, outlayer_p);
    
    EXPECT_EQ(inlayer_p->outgoing_n, 1) << "Inmod number not correct"; 
    EXPECT_EQ(outlayer_p->incoming_n, 1) << "Outmod number not correct";
    
    Connection& con = inlayer_p->outgoing_p[0];
    Connection& con_ = outlayer_p->incoming_p[0];
    
    EXPECT_EQ((void*) con.inlayer_p, (void*) con_.inlayer_p)
        << "Different inlayer.";
    EXPECT_EQ((void*) con.outlayer_p, (void*) con_.outlayer_p)
        << "Different outlayer.";
    EXPECT_EQ(con.type, 0) 
        << "Wrong type.";
    EXPECT_EQ(con_.type, 0) 
        << "Wrong type.";
    EXPECT_EQ(con.recurrent, con_.recurrent) 
        << "Different recurrent flag.";
    EXPECT_EQ(con.inlayerstart, con_.inlayerstart) 
        << "Different inlayerstart.";
    EXPECT_EQ(con.inlayerstop, con_.inlayerstop) 
        << "Different inlayerstop.";
    EXPECT_EQ(con.outlayerstart, con_.outlayerstart) 
        << "Different outlayerstart.";
    EXPECT_EQ(con.outlayerstop, con_.outlayerstop) 
        << "Different outlayerstop.";
        
    // free(inlayer_p);
    // free(outlayer_p);
}


TEST(TestBuildNetwork, FullConnectionEquality) {
    Layer* inlayer_p = make_layer(2, 2);
    Layer* outlayer_p = make_layer(2, 2);
    
    double weights[] = {0, 0, 0, 0};
    
    full_connect(inlayer_p, outlayer_p, weights);
    
    EXPECT_EQ(inlayer_p->outgoing_n, 1) 
        << "Inmod number not correct"; 
    EXPECT_EQ(outlayer_p->incoming_n, 1) 
        << "Outmod number not correct";
    
    Connection& con = inlayer_p->outgoing_p[0];
    Connection& con_ = outlayer_p->incoming_p[0];
    
    EXPECT_EQ((void*) con.inlayer_p, (void*) con_.inlayer_p)
        << "Different inlayer.";
    EXPECT_EQ((void*) con.outlayer_p, (void*) con_.outlayer_p)
        << "Different outlayer.";
    EXPECT_EQ(con.type, 1) 
        << "Wrong type.";
    EXPECT_EQ(con_.type, 1) 
        << "Wrong type.";
    EXPECT_EQ(con.internal.full_connection_p, con_.internal.full_connection_p)
        << "Don't refer to same FullConnection";
    EXPECT_EQ(con.recurrent, con_.recurrent) 
        << "Different recurrent flag.";
    EXPECT_EQ(con.inlayerstart, con_.inlayerstart) 
        << "Different inlayerstart.";
    EXPECT_EQ(con.inlayerstop, con_.inlayerstop) 
        << "Different inlayerstop.";
    EXPECT_EQ(con.outlayerstart, con_.outlayerstart) 
        << "Different outlayerstart.";
    EXPECT_EQ(con.outlayerstop, con_.outlayerstop) 
        << "Different outlayerstop.";
        
    // free(inlayer_p);
    // free(outlayer_p);
}


TEST(TestBuildNetwork, AppendConnection) {
    Layer* layer_p = make_layer(1, 1);
    
    Connection* con_p = (Connection*) malloc(sizeof(Connection) * 2);
    
    con_p[0].inlayer_p = (Layer*) 0;
    con_p[0].outlayer_p = (Layer*) 1;
    con_p[0].type = 0;

    con_p[1].inlayer_p = (Layer*) 2;
    con_p[1].outlayer_p = (Layer*) 3;
    con_p[1].type = 1;

    EXPECT_EQ(layer_p->outgoing_n, 0);
    
    append_to_array(layer_p->outgoing_p, layer_p->outgoing_n, con_p[0]);
    free(con_p);
    
    EXPECT_EQ(layer_p->outgoing_n, 1);
    
    EXPECT_EQ(layer_p->outgoing_p[0].inlayer_p, (Layer*) 0);
    EXPECT_EQ(layer_p->outgoing_p[0].outlayer_p, (Layer*) 1);
    EXPECT_EQ(layer_p->outgoing_p[0].type, 0);
    
    append_to_array(layer_p->outgoing_p, layer_p->outgoing_n, con_p[1]);
    
    EXPECT_EQ(layer_p->outgoing_n, 2);
    
    EXPECT_EQ(layer_p->outgoing_p[0].inlayer_p, (Layer*) 0);
    EXPECT_EQ(layer_p->outgoing_p[0].outlayer_p, (Layer*) 1);
    EXPECT_EQ(layer_p->outgoing_p[0].type, 0);

    EXPECT_EQ(layer_p->outgoing_p[1].inlayer_p, (Layer*) 2);
    EXPECT_EQ(layer_p->outgoing_p[1].outlayer_p, (Layer*) 3);
    EXPECT_EQ(layer_p->outgoing_p[1].type, 1);
}


TEST(TestForwardPass, TestSimpleIdentityConnectionPass) {
    Layer* layers_p = (Layer*) malloc(sizeof(Layer) * 2);
    make_linear_layer(layers_p, 1);
    make_linear_layer(layers_p + 1, 1);
    identity_connect(layers_p, layers_p + 1);
    
    layers_p[0].inputs.contents_p[0] = 1.0;
    
    activate(layers_p, 2);
    
    EXPECT_EQ(layers_p[0].inputs.contents_p[0], 1.0);
    EXPECT_EQ(layers_p[0].outputs.contents_p[0], 1.0);
    EXPECT_EQ(layers_p[1].inputs.contents_p[0], 1.0);
    EXPECT_EQ(layers_p[1].outputs.contents_p[0], 1.0);
    
    // free(layers_p);
}


TEST(TestForwardPass, TestSimpleSigmoidLayerPass) {
    Layer* layer_p = make_sigmoid_layer(5);
    double input[] = {-1, -0.5, 0.0, 0.5, 1.0};
    layer_p->inputs.contents_p = input;
    activate(layer_p, 1);

    EXPECT_TRUE(abs(layer_p->outputs.contents_p[0] - 0.26894) < 0.001);
    EXPECT_TRUE(abs(layer_p->outputs.contents_p[1] - 0.37754) < 0.001);
    EXPECT_TRUE(abs(layer_p->outputs.contents_p[2] - 0.5) < 0.001);  
    EXPECT_TRUE(abs(layer_p->outputs.contents_p[3] - 0.62245) < 0.001);
    EXPECT_TRUE(abs(layer_p->outputs.contents_p[4] - 0.73105) < 0.001);
    
    // free(layer_p);
}


TEST(TestForwardPass, TestBiasLayerPass) {
    Layer* layer_p = make_bias_layer();
    activate(layer_p, 1);
    EXPECT_TRUE(layer_p->outputs.contents_p[0] == 1.0);
    
    // free(layer_p);
}


TEST(TestForwardPass, TestMdLstmLayerPass)
{
    Layer* layer_p = make_mdlstm_layer(2, 1);
    double input[] = {-2, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    layer_p->inputs.contents_p = input;
    activate(layer_p, 1);
    
    EXPECT_TRUE(abs(layer_p->outputs.contents_p[0] -  1.99505288) < 0.0001)
        << layer_p->outputs.contents_p[0] << " should be " << "1.99505288";
    EXPECT_TRUE(abs(layer_p->outputs.contents_p[1] -  1.99817789) < 0.0001)
        << layer_p->outputs.contents_p[1] << " should be " << "1.99817789";
    EXPECT_TRUE(abs(layer_p->outputs.contents_p[2] -  7.28462257) < 0.0001)
        << layer_p->outputs.contents_p[2] << " should be " << "7.28462257";
    EXPECT_TRUE(abs(layer_p->outputs.contents_p[3] -  10.03515154) < 0.0001)
        << layer_p->outputs.contents_p[3] << " should be " << "10.03515154";
}


TEST(TestBackwardPass, TestMdLstmLayerPass)
{
    Layer* layer_p = make_mdlstm_layer(2, 1);
    double input[] = {-2, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    double error[] = {-1, 0, 1, 2};
    layer_p->inputs.contents_p = input;
    activate(layer_p, 1);
    layer_p->outputs.error_p = error;
    calc_derivs(layer_p, 1);
    
    EXPECT_TRUE(abs(layer_p->inputs.error_p[0] - 0.10492) < 0.0001)
            << layer_p->inputs.error_p[0] << " should be " << "0.629539785";

    EXPECT_TRUE(abs(layer_p->inputs.error_p[1] - 0.39319) < 0.0001)
            << layer_p->inputs.error_p[1] << " should be " << "1.572752666";

    EXPECT_TRUE(abs(layer_p->inputs.error_p[2] - 0.83995) < 0.0001)
            << layer_p->inputs.error_p[2] << " should be " << "2.519849204";

    EXPECT_TRUE(abs(layer_p->inputs.error_p[3] - 0.81318) < 0.0001)
            << layer_p->inputs.error_p[3] << " should be " << "1.626359763";

    EXPECT_TRUE(abs(layer_p->inputs.error_p[4] - 0.00016) < 0.0001)
            << layer_p->inputs.error_p[4] << " should be " << "0.000959073";

    EXPECT_TRUE(abs(layer_p->inputs.error_p[5] - 0.00027) < 0.0001)
            << layer_p->inputs.error_p[5] << " should be " << "0.001061984";

    EXPECT_TRUE(abs(layer_p->inputs.error_p[6] - -0.00247) < 0.0001)
            << layer_p->inputs.error_p[6] << " should be " << "0.004933014";

    EXPECT_TRUE(abs(layer_p->inputs.error_p[7] - 0) < 0.0001)
            << layer_p->inputs.error_p[7] << " should be " << "0.003640885";

    EXPECT_TRUE(abs(layer_p->inputs.error_p[8] - 0.88079) < 0.0001)
            << layer_p->inputs.error_p[8] << " should be " << "2.642394542";

    EXPECT_TRUE(abs(layer_p->inputs.error_p[9] - 1.90515) < 0.0001)
            << layer_p->inputs.error_p[9] << " should be " << "3.810296537";

}


TEST(TestForwardPass, TestLstmLayerPass)
{
    Layer* layer_p = make_lstm_layer(2);
    int time_p[] = {0};
    layer_p->timestep_p = time_p;
    
    double input[] = {-2, 1, 2, 3, 4, 5, 6, 7};
    layer_p->inputs.contents_p = input;
    activate(layer_p, 1);
    
    EXPECT_TRUE(abs(layer_p->outputs.contents_p[0] -  0.11827) < 0.0001)
        << layer_p->outputs.contents_p[0] << " should be " << "1.99505288";
    EXPECT_TRUE(abs(layer_p->outputs.contents_p[1] -  0.62310) < 0.0001)
        << layer_p->outputs.contents_p[1] << " should be " << "1.99817789";
}


TEST(TestBackwardPass, TestLstmLayerPass)
{
    Layer* layer_p = make_lstm_layer(2);
    double input[] = {-2, 1, 2, 3, 4, 5, 6, 7};
    double error[] = {-1, 0, 1, 2};
    int time_p[] = {0};
    int time_p2[] = {1};
    layer_p->timestep_p = time_p;
    layer_p->inputs.contents_p = input;
    activate(layer_p, 1);
    layer_p->timestep_p = time_p2;
    layer_p->outputs.error_p = error;
    calc_derivs(layer_p, 1);
    
    EXPECT_TRUE(abs(layer_p->inputs.error_p[0] - -0.19408) < 0.0001)
            << layer_p->inputs.error_p[0] << " should be " << "0.10492";

    EXPECT_TRUE(abs(layer_p->inputs.error_p[1] - 0.11717) < 0.0001)
            << layer_p->inputs.error_p[1] << " should be " << "0.39319";

    EXPECT_TRUE(abs(layer_p->inputs.error_p[2] - 0.0) < 0.0001)
            << layer_p->inputs.error_p[2] << " should be " << "0.83995";

    EXPECT_TRUE(abs(layer_p->inputs.error_p[3] - 0.0) < 0.0001)
            << layer_p->inputs.error_p[3] << " should be " << "0.81318";

    EXPECT_TRUE(abs(layer_p->inputs.error_p[4] - -0.00030) < 0.0001)
            << layer_p->inputs.error_p[4] << " should be " << "0.00016";

    EXPECT_TRUE(abs(layer_p->inputs.error_p[5] - 0.00008) < 0.0001)
            << layer_p->inputs.error_p[5] << " should be " << "0.00027";

    EXPECT_TRUE(abs(layer_p->inputs.error_p[6] - -0.00029) < 0.0001)
            << layer_p->inputs.error_p[6] << " should be " << "0.00247";

    EXPECT_TRUE(abs(layer_p->inputs.error_p[7] - 0.00028) < 0.0001)
            << layer_p->inputs.error_p[7] << " should be " << "0.0";
}


TEST(TestForwardPass, TestForkedIdentityPass) {
    Layer* layers_p = (Layer*) malloc(sizeof(Layer) * 4);

    make_linear_layer(layers_p, 1);
    ASSERT_EQ(layers_p[0].inputs.size, 1);
    ASSERT_EQ(layers_p[0].outputs.size, 1);
    
    make_linear_layer(layers_p + 1, 1);
    ASSERT_EQ(layers_p[1].inputs.size, 1);
    ASSERT_EQ(layers_p[1].outputs.size, 1);

    make_linear_layer(layers_p + 2, 1);
    ASSERT_EQ(layers_p[2].inputs.size, 1);
    ASSERT_EQ(layers_p[2].outputs.size, 1);
    
    make_linear_layer(layers_p + 3, 1);
    ASSERT_EQ(layers_p[3].inputs.size, 1);
    ASSERT_EQ(layers_p[3].outputs.size, 1);
    
    identity_connect(layers_p, layers_p + 1);
    ASSERT_EQ(layers_p[0].outgoing_p[0].inlayer_p, layers_p);

    identity_connect(layers_p, layers_p + 2);
    ASSERT_EQ(layers_p[0].outgoing_p[1].inlayer_p, layers_p);

    identity_connect(layers_p + 1, layers_p + 3);
    ASSERT_EQ(layers_p[1].outgoing_p[0].inlayer_p, layers_p + 1);
    
    identity_connect(layers_p + 2, layers_p + 3);
    ASSERT_EQ(layers_p[2].outgoing_p[0].inlayer_p, layers_p + 2);

    ASSERT_EQ(layers_p[0].inputs.size, 1);
    ASSERT_EQ(layers_p[0].outputs.size, 1);
    ASSERT_EQ(layers_p[1].inputs.size, 1);
    ASSERT_EQ(layers_p[1].outputs.size, 1);
    ASSERT_EQ(layers_p[2].inputs.size, 1);
    ASSERT_EQ(layers_p[2].outputs.size, 1);
    ASSERT_EQ(layers_p[3].inputs.size, 1);
    ASSERT_EQ(layers_p[3].outputs.size, 1);
    
    layers_p[0].inputs.contents_p[0] = 1.5;
    activate(layers_p, 4);

    EXPECT_EQ(layers_p[3].inputs.contents_p[0], 3.0);
    EXPECT_EQ(layers_p[3].outputs.contents_p[0], 3.0);
    
    // free(layers_p);
}


TEST(TestForwardPass, TestFullConnectionPass1) {
    Layer* layers_p = (Layer*) malloc(sizeof(Layer) * 4);
    
    Layer* inlayer_p = layers_p;
    make_linear_layer(inlayer_p, 3);
    
    Layer* outlayer_p = layers_p + 1;
    make_linear_layer(outlayer_p, 2);
    
    double weights[] = {2.5, 3.0, 1.0, 3.0, 4.0, -3.0};

    full_connect(inlayer_p, outlayer_p, weights);
    
    double input[] = {1.2, -2.25, 5.0};
    inlayer_p->inputs.contents_p = input;

    activate(layers_p, 2);
    
    EXPECT_EQ(outlayer_p->outputs.contents_p[0], 1.25);
    EXPECT_EQ(outlayer_p->outputs.contents_p[1], -20.4);
    
    // free(layers_p);
}


TEST(TestForwardPass, TestFullConnectionPass2) {
    Layer* layers_p = (Layer*) malloc(sizeof(Layer) * 4);
    
    Layer* inlayer_p = layers_p;
    make_linear_layer(inlayer_p, 2);
    
    Layer* outlayer_p = layers_p + 1;
    make_linear_layer(outlayer_p, 3);
    
    double weights[] = {2.5, 3.0, 1.0, 3.0, 4.0, -3.0};

    full_connect(inlayer_p, outlayer_p, weights);
    
    double input[] = {1.2, -2.25};
    inlayer_p->inputs.contents_p = input;

    activate(layers_p, 2);
    
    EXPECT_EQ(outlayer_p->outputs.contents_p[0], -3.75);
    EXPECT_EQ(outlayer_p->outputs.contents_p[1], -5.55);
    EXPECT_EQ(outlayer_p->outputs.contents_p[2], 11.55);
    
    // free(layers_p);
}


TEST(TestForwardPass, TestSigmoidIdentityLayerConnected)
{
    Layer* layers_p = (Layer*) malloc(sizeof(Layer) * 2);
    make_sigmoid_layer(layers_p, 3);
    make_linear_layer(layers_p + 1, 3);
    identity_connect(layers_p, layers_p + 1);

    double inputs[] = {-1, -0.5, 0.0};
    
    layers_p->inputs.contents_p = inputs;
    activate(layers_p, 2);
    
    EXPECT_TRUE(abs(layers_p[1].outputs.contents_p[0] - 0.3678) < 0.001);
    EXPECT_TRUE(abs(layers_p[1].outputs.contents_p[1] - 0.6065) < 0.001);
    EXPECT_TRUE(abs(layers_p[1].outputs.contents_p[2] - 1.0) < 0.001);
    
    // free(layers_p);
}


TEST(TestForwardPass, TestSlicedIdentityConnection)
{
    Layer* layers_p = (Layer*) malloc(sizeof(Layer) * 3);
    Layer* inlayer_p = layers_p;
    make_linear_layer(inlayer_p, 6);
    
    Layer* outlayer_1st_half_p = layers_p + 1;
    make_linear_layer(outlayer_1st_half_p, 3);
    
    Layer* outlayer_2nd_half_p = layers_p + 2;
    make_linear_layer(outlayer_2nd_half_p, 3);
    
    identity_connect(inlayer_p, outlayer_1st_half_p, 0, 3, 0, 3);
    identity_connect(inlayer_p, outlayer_2nd_half_p, 3, 6, 0, 3);
    
    setTimestepPointer(layers_p + 1, 2, layers_p->timestep_p);
    
    double inputs[] = {1, 2, 3, 4, 5, 6};
    inlayer_p->inputs.contents_p = inputs;
    
    activate(layers_p, 3);
    
    EXPECT_EQ(layers_p[1].outputs.contents_p[0], 1);
    EXPECT_EQ(layers_p[1].outputs.contents_p[1], 2);
    EXPECT_EQ(layers_p[1].outputs.contents_p[2], 3);
    EXPECT_EQ(layers_p[2].outputs.contents_p[0], 4);
    EXPECT_EQ(layers_p[2].outputs.contents_p[1], 5);
    EXPECT_EQ(layers_p[2].outputs.contents_p[2], 6);
    
    // free(layers_p);
}

TEST(TestForwardPass, TestSlicedFullConnection)
{
    Layer* layers_p = (Layer*) malloc(sizeof(Layer) * 3);
    Layer* inlayer_p = layers_p;
    make_linear_layer(inlayer_p, 6);
    
    Layer* outlayer_1st_half_p = layers_p + 1;
    make_linear_layer(outlayer_1st_half_p, 3);
    
    Layer* outlayer_2nd_half_p = layers_p + 2;
    make_linear_layer(outlayer_2nd_half_p, 3);
    
    double weights[] = {1, 0, 0,
                          0, 2, 0,
                          0, 0, 0};
                         
    full_connect(inlayer_p, outlayer_1st_half_p, weights, 0, 3, 0, 3);
    full_connect(inlayer_p, outlayer_2nd_half_p, weights, 3, 6, 0, 3);
    
    setTimestepPointer(layers_p + 1, 2, layers_p->timestep_p);
    
    double inputs[] = {1, 2, 3, 4, 5, 6};
    inlayer_p->inputs.contents_p = inputs;
    
    activate(layers_p, 3);
    
    EXPECT_EQ(layers_p[1].outputs.contents_p[0], 1);
    EXPECT_EQ(layers_p[1].outputs.contents_p[1], 4);
    EXPECT_EQ(layers_p[1].outputs.contents_p[2], 0);
    EXPECT_EQ(layers_p[2].outputs.contents_p[0], 4);
    EXPECT_EQ(layers_p[2].outputs.contents_p[1], 10);
    EXPECT_EQ(layers_p[2].outputs.contents_p[2], 0);
    
    // free(layers_p);
}


TEST(TestBackwardPass, TestIdentityLayer)
{
    Layer* layer_p = make_linear_layer(3);
    
    double inputs[] = {2, 0, 1};
    double errors[] = {2, 0, 1};

    layer_p->inputs.contents_p = inputs;
    layer_p->outputs.error_p = errors;
    
    activate(layer_p, 1);
    calc_derivs(layer_p, 1);
    
    EXPECT_EQ(layer_p->inputs.error_p[0], 2);
    EXPECT_EQ(layer_p->inputs.error_p[1], 0);
    EXPECT_EQ(layer_p->inputs.error_p[2], 1);
    
    // free(layer_p);
}


TEST(TestBackwardPass, TestSigmoidLayer)
{
    Layer* layer_p = make_sigmoid_layer(3);
    
    double inputs[] = {-2, 0.5, 2};
    double errors[] = {2, 3, 1};
    
    layer_p->inputs.contents_p = inputs;
    activate(layer_p, 1);
    layer_p->outputs.error_p = errors;
    calc_derivs(layer_p, 1);
    
    EXPECT_TRUE(abs(layer_p->inputs.error_p[0] - 0.209987) < 0.0001)
        << layer_p->inputs.error_p[0] << " should be " << "0.209987";
    EXPECT_TRUE(abs(layer_p->inputs.error_p[1] - 0.705011) < 0.0001)
        << layer_p->inputs.error_p[1] << " should be " << "0.705011";
    EXPECT_TRUE(abs(layer_p->inputs.error_p[2] - 0.104994) < 0.0001)
        << layer_p->inputs.error_p[2] << " should be " << "0.104994";
    
    // free(layer_p);
}


TEST(TestBackwardPass, TestIdentityConnection)
{
    Layer* layers_p = (Layer*) malloc(sizeof(Layer) * 2);
    make_linear_layer(layers_p, 3);
    make_linear_layer(layers_p + 1, 3);
    identity_connect(layers_p, layers_p + 1);

    double inputs[] = {-2, 0.5, 2};
    double errors[] = {2, 3, 1};
    
    setTimestepPointer(layers_p + 1, 1, layers_p->timestep_p);
    
    layers_p->inputs.contents_p = inputs;
    activate(layers_p, 2);
    layers_p[1].outputs.error_p = errors;
    calc_derivs(layers_p, 2);
    
    EXPECT_EQ(layers_p[1].outputs.error_p[0], 2);
    EXPECT_EQ(layers_p[1].outputs.error_p[1], 3);
    EXPECT_EQ(layers_p[1].outputs.error_p[2], 1);
    EXPECT_EQ(layers_p[1].inputs.error_p[0], 2);
    EXPECT_EQ(layers_p[1].inputs.error_p[1], 3);
    EXPECT_EQ(layers_p[1].inputs.error_p[2], 1);
    EXPECT_EQ(layers_p[0].outputs.error_p[0], 2);
    EXPECT_EQ(layers_p[0].outputs.error_p[1], 3);
    EXPECT_EQ(layers_p[0].outputs.error_p[2], 1);
    EXPECT_EQ(layers_p[0].inputs.error_p[0], 2);
    EXPECT_EQ(layers_p[0].inputs.error_p[1], 3);
    EXPECT_EQ(layers_p[0].inputs.error_p[2], 1);
    
    // free(layers_p);
}


TEST(TestBackwardPass, TestFullConnection)
{
    Layer* layers_p = (Layer*) malloc(sizeof(Layer) * 2);
    make_linear_layer(layers_p, 2);
    make_linear_layer(layers_p + 1, 3);
    
    setTimestepPointer(layers_p + 1, 1, layers_p->timestep_p);
    
    double inputs[] = {-1, 2};
    double weights[] = {1., 2., 3., 4., 5., 0.};
    
    layers_p->inputs.contents_p = inputs;
    
    full_connect(layers_p, layers_p + 1, weights);
    
    activate(layers_p, 2);
    
    layers_p[1].outputs.error_p[0] = -8;
    layers_p[1].outputs.error_p[1] = -7;
    layers_p[1].outputs.error_p[2] = 2;
    calc_derivs(layers_p, 2);
    
    Connection* con_p = layers_p->outgoing_p;
    
    EXPECT_EQ(con_p->internal.full_connection_p->weights.error_p[0], 8);
    EXPECT_EQ(con_p->internal.full_connection_p->weights.error_p[1], -16);
    EXPECT_EQ(con_p->internal.full_connection_p->weights.error_p[2], 7);
    EXPECT_EQ(con_p->internal.full_connection_p->weights.error_p[3], -14);
    EXPECT_EQ(con_p->internal.full_connection_p->weights.error_p[4], -2);
    EXPECT_EQ(con_p->internal.full_connection_p->weights.error_p[5], 4);
    
    // free(layers_p);
}

}  // namespace


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}