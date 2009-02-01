#include <gtest/gtest.h>
#include <iostream>
#include "../arac.h"

namespace AracTesting {


using namespace arac::common;
using namespace arac::structure::modules;
using namespace arac::structure::connections;
using namespace arac::structure::networks;
using namespace arac::structure::networks::mdrnns;


TEST(TestCommon, TestBuffer) {
    Buffer buffer = Buffer(2);
    double* addend_p = new double[2];
    addend_p[0] = 1.2;
    addend_p[1] = 2.4;
    
    EXPECT_DOUBLE_EQ(0, buffer[0][0])
        << "Buffer not correctly initialized";
    EXPECT_DOUBLE_EQ(0, buffer[0][1])
        << "Buffer not correctly initialized";
    
    buffer.add(addend_p);
    
    EXPECT_DOUBLE_EQ(1.2, buffer[0][0])
        << "Adding to buffer incorrect.";
    EXPECT_DOUBLE_EQ(2.4, buffer[0][1])
        << "Adding to buffer incorrect.";
    
    buffer.add(addend_p);
    
    EXPECT_DOUBLE_EQ(2.4, buffer[0][0])
        << "Adding to buffer incorrect.";
    EXPECT_DOUBLE_EQ(4.8, buffer[0][1])
        << "Adding to buffer incorrect.";
        
    buffer.expand();
    
    EXPECT_DOUBLE_EQ(0, buffer[1][0])
        << "Buffer not correctly initialized";
    EXPECT_DOUBLE_EQ(0, buffer[1][1])
        << "Buffer not correctly initialized";
        
    EXPECT_EQ(2, buffer.size())
        << "Buffersize incorrect.";
        
    buffer.add(addend_p);
    
    EXPECT_DOUBLE_EQ(1.2, buffer[1][0])
        << "Adding to buffer incorrect.";
    EXPECT_DOUBLE_EQ(2.4, buffer[1][1])
        << "Adding to buffer incorrect.";

    buffer.clear();
    
    EXPECT_DOUBLE_EQ(0, buffer[0][0])
        << "Setting buffer to zero incorrect.";
    EXPECT_DOUBLE_EQ(0, buffer[0][1])
        << "Setting buffer to zero incorrect.";
    EXPECT_DOUBLE_EQ(0, buffer[1][0])
        << "Setting buffer to zero incorrect.";
    EXPECT_DOUBLE_EQ(0, buffer[1][1])
        << "Setting buffer to zero incorrect.";
}


TEST(TestCommon, TestBufferMemory) {
    Buffer* buffer_p = new Buffer(2);
    buffer_p->free_memory();
}


TEST(TestCommon, TestBufferClear) {
    Buffer buffer(2);
    double* content_p = new double[2];
    content_p[0] = 1;
    content_p[1] = 1;
    buffer.add(content_p);
    buffer.expand();
    buffer.add(content_p);
    EXPECT_DOUBLE_EQ(1, buffer[0][0])
        << "Adding buffer values did not work.";
    EXPECT_DOUBLE_EQ(1, buffer[0][1])
        << "Adding buffer values did not work.";
    EXPECT_DOUBLE_EQ(1, buffer[1][0])
        << "Adding buffer values did not work.";
    EXPECT_DOUBLE_EQ(1, buffer[1][1])
        << "Adding buffer values did not work.";
    buffer.clear_at(0);
    EXPECT_DOUBLE_EQ(0, buffer[0][0])
        << "Clearing buffer incorrect.";
    EXPECT_DOUBLE_EQ(0, buffer[0][1])
        << "Clearing buffer incorrect.";
    buffer.clear();
    EXPECT_DOUBLE_EQ(0, buffer[0][0])
        << "Clearing buffer incorrect.";
    EXPECT_DOUBLE_EQ(0, buffer[0][1])
        << "Clearing buffer incorrect.";
    EXPECT_DOUBLE_EQ(0, buffer[1][0])
        << "Clearing buffer incorrect.";
    EXPECT_DOUBLE_EQ(0, buffer[1][1])
        << "Clearing buffer incorrect.";
}


TEST(TestModules, TestModuleClear) {
    LinearLayer layer(2);
    Buffer& buffer = layer.input();
    double* content_p = new double[2];
    content_p[0] = 1;
    content_p[1] = 1;
    
    layer.add_to_input(content_p);
    EXPECT_DOUBLE_EQ(1, buffer[0][0])
        << "Adding values did not work.";
    EXPECT_DOUBLE_EQ(1, buffer[0][1])
        << "Adding values did not work.";
        
    layer.clear();
    
    EXPECT_DOUBLE_EQ(0, buffer[0][0])
        << "Clearing Module incorrect.";
    EXPECT_DOUBLE_EQ(0, buffer[0][1])
        << "Clearing Module incorrect.";
}


TEST(TestModules, BiasUnit) {
    Bias* bias_p = new Bias();

    bias_p->forward();

    ASSERT_DOUBLE_EQ(1, bias_p->output()[0][0])
        << "Bias forward not working.";

    bias_p->forward();

    ASSERT_DOUBLE_EQ(1, bias_p->output()[0][0])
        << "Bias forward not working.";
}


TEST(TestModules, LinearLayer) {
    LinearLayer* layer_p = new LinearLayer(2);

    double* input_p = new double[2];
    input_p[0] = 2.;
    input_p[1] = 3.;

    ASSERT_DOUBLE_EQ(0, layer_p->input()[0][0])
        << "LinearLayer::add_to_input not working.";
    ASSERT_DOUBLE_EQ(0, layer_p->input()[0][1])
        << "LinearLayer::add_to_input not working.";

    layer_p->add_to_input(input_p);
    
    ASSERT_DOUBLE_EQ(2, layer_p->input()[0][0])
        << "LinearLayer::add_to_input not working.";
    ASSERT_DOUBLE_EQ(3, layer_p->input()[0][1])
        << "LinearLayer::add_to_input not working.";
    
    layer_p->forward();
    
    ASSERT_DOUBLE_EQ(2, layer_p->output()[0][0])
        << "Forward pass incorrect.";
        
    ASSERT_DOUBLE_EQ(3, layer_p->output()[0][1])
        << "Forward pass incorrect.";

    double* outerror_p = new double[2];
    outerror_p[0] = 1;
    outerror_p[1] = -.3;
    layer_p->add_to_outerror(outerror_p);
    layer_p->backward();
    
    EXPECT_DOUBLE_EQ(1, layer_p->inerror()[0][0])
        << "Backward pass incorrect.";
    
    EXPECT_DOUBLE_EQ(-0.3, layer_p->inerror()[0][1])
        << "Backward pass incorrect.";
}


TEST(TestModules, LinearLayerSequential) {
    LinearLayer* layer_p = new LinearLayer(2);
    layer_p->set_mode(Component::Sequential);

    double* input_p = new double[2];
    input_p[0] = 2.;
    input_p[1] = 3.;

    layer_p->add_to_input(input_p);
    
    ASSERT_DOUBLE_EQ(2, layer_p->input()[0][0])
        << "LinearLayer::add_to_input not working.";
    ASSERT_DOUBLE_EQ(3, layer_p->input()[0][1])
        << "LinearLayer::add_to_input not working.";
    
    layer_p->forward();

    ASSERT_DOUBLE_EQ(2, layer_p->output()[0][0])
        << "Forward pass incorrect.";
        
    ASSERT_DOUBLE_EQ(3, layer_p->output()[0][1])
        << "Forward pass incorrect.";
    
    layer_p->add_to_input(input_p);
    layer_p->add_to_input(input_p);
    
    ASSERT_DOUBLE_EQ(4, layer_p->input()[1][0])
        << "LinearLayer::add_to_input not working.";
    ASSERT_DOUBLE_EQ(6, layer_p->input()[1][1])
        << "LinearLayer::add_to_input not working.";
        
    layer_p->forward();
    
    ASSERT_DOUBLE_EQ(4, layer_p->output()[1][0])
        << "Forward pass incorrect.";
        
    ASSERT_DOUBLE_EQ(6, layer_p->output()[1][1])
        << "Forward pass incorrect.";

    double* outerror_p = new double[2];
    outerror_p[0] = 1;
    outerror_p[1] = -.3;

    layer_p->add_to_outerror(outerror_p);
    
    EXPECT_DOUBLE_EQ(1, layer_p->outerror()[1][0])
        << "Backward pass incorrect.";
    
    EXPECT_DOUBLE_EQ(-0.3, layer_p->outerror()[1][1])
        << "Backward pass incorrect.";
    
    layer_p->backward();
    
    EXPECT_DOUBLE_EQ(1, layer_p->inerror()[1][0])
        << "Backward pass incorrect.";
    
    EXPECT_DOUBLE_EQ(-0.3, layer_p->inerror()[1][1])
        << "Backward pass incorrect.";
        
    layer_p->add_to_outerror(outerror_p);
    layer_p->add_to_outerror(outerror_p);

    EXPECT_DOUBLE_EQ(1, layer_p->outerror()[1][0])
        << "Backward pass incorrect.";
    
    EXPECT_DOUBLE_EQ(-0.3, layer_p->outerror()[1][1])
        << "Backward pass incorrect.";
    
    layer_p->backward();
    
    EXPECT_DOUBLE_EQ(2, layer_p->inerror()[0][0])
        << "Backward pass incorrect.";
    
    EXPECT_DOUBLE_EQ(-0.6, layer_p->inerror()[0][1])
        << "Backward pass incorrect.";
}


TEST(TestModules, SigmoidLayer) {
    SigmoidLayer* layer_p = new SigmoidLayer(5);
    
    double* input_p = new double[5];
    input_p[0] = -1;
    input_p[1] = -0.5;
    input_p[2] = 0;
    input_p[3] = 0.5;
    input_p[4] = 1;
    
    layer_p->add_to_input(input_p);
    
    ASSERT_DOUBLE_EQ(-1, layer_p->input()[0][0])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(-0.5, layer_p->input()[0][1])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(0, layer_p->input()[0][2])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(0.5, layer_p->input()[0][3])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(1, layer_p->input()[0][4])
        << "add_to_input not working.";
    
    layer_p->forward();
    
    EXPECT_DOUBLE_EQ(0.2689414213699951, layer_p->output()[0][0])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.37754066879814541, layer_p->output()[0][1])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.5, layer_p->output()[0][2])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.62245933120185459, layer_p->output()[0][3])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.7310585786300049, layer_p->output()[0][4])
        << "Forward pass incorrect.";
    
    double* outerror_p = new double[5];
    outerror_p[0] = 2;
    outerror_p[1] = 4;
    outerror_p[2] = 6;
    outerror_p[3] = 8;
    outerror_p[4] = 10;
    
    layer_p->add_to_outerror(outerror_p);
    layer_p->backward();
    
    EXPECT_DOUBLE_EQ(0.3932238664829637, layer_p->inerror()[0][0])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.94001484880637798, layer_p->inerror()[0][1])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(1.5, layer_p->inerror()[0][2])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(1.880029697612756, layer_p->inerror()[0][3])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(1.9661193324148185, layer_p->inerror()[0][4])
        << "Backward pass incorrect.";
}


TEST(TestModules, TanhLayer) {
    TanhLayer* layer_p = new TanhLayer(5);
    
    double* input_p = new double[5];
    input_p[0] = -1;
    input_p[1] = -0.5;
    input_p[2] = 0;
    input_p[3] = 0.5;
    input_p[4] = 1;
    
    layer_p->add_to_input(input_p);
    
    EXPECT_DOUBLE_EQ(-1, layer_p->input()[0][0])
        << "add_to_input not working.";
    EXPECT_DOUBLE_EQ(-0.5, layer_p->input()[0][1])
        << "add_to_input not working.";
    EXPECT_DOUBLE_EQ(0, layer_p->input()[0][2])
        << "add_to_input not working.";
    EXPECT_DOUBLE_EQ(0.5, layer_p->input()[0][3])
        << "add_to_input not working.";
    EXPECT_DOUBLE_EQ(1, layer_p->input()[0][4])
        << "add_to_input not working.";
    
    layer_p->forward();
    
    EXPECT_DOUBLE_EQ(-0.76159415595576485, layer_p->output()[0][0])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(-0.46211715726000974, layer_p->output()[0][1])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0, layer_p->output()[0][2])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.46211715726000974, layer_p->output()[0][3])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.76159415595576485, layer_p->output()[0][4])
        << "Forward pass incorrect.";
    
    double* outerror_p = new double[5];
    outerror_p[0] = 2;
    outerror_p[1] = 4;
    outerror_p[2] = 6;
    outerror_p[3] = 8;
    outerror_p[4] = 10;
    
    layer_p->add_to_outerror(outerror_p);
    layer_p->backward();
    
    EXPECT_DOUBLE_EQ(0.83994868322805227, layer_p->inerror()[0][0])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(3.1457909318637096, layer_p->inerror()[0][1])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(6, layer_p->inerror()[0][2])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(6.2915818637274192, layer_p->inerror()[0][3])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(4.1997434161402616, layer_p->inerror()[0][4])
        << "Backward pass incorrect.";
}


TEST(TestModules, SoftmaxLayer) {
    SoftmaxLayer* layer_p = new SoftmaxLayer(2);

    double* input_p = new double[2];
    input_p[0] = 2.;
    input_p[1] = 4.;

    layer_p->add_to_input(input_p);
    
    ASSERT_DOUBLE_EQ(2, layer_p->input()[0][0])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(4, layer_p->input()[0][1])
        << "add_to_input not working.";
    
    layer_p->forward();
    
    ASSERT_DOUBLE_EQ(0.11920292202211756, layer_p->output()[0][0])
        << "Forward pass incorrect.";
        
    ASSERT_DOUBLE_EQ(0.88079707797788243, layer_p->output()[0][1])
        << "Forward pass incorrect.";

    double* outerror_p = new double[2];
    outerror_p[0] = 2;
    outerror_p[1] = 4;
    layer_p->add_to_outerror(outerror_p);
    layer_p->backward();
    
    EXPECT_DOUBLE_EQ(2, layer_p->inerror()[0][0])
        << "Backward pass incorrect.";
    
    EXPECT_DOUBLE_EQ(4, layer_p->inerror()[0][1])
        << "Backward pass incorrect.";
}


TEST(TestModules, GateLayer) {
    GateLayer* layer_p = new GateLayer(2);
    
    double* input_p = new double[4];
    input_p[0] = 1;
    input_p[1] = 2;
    input_p[2] = 3;
    input_p[3] = 4;
    
    layer_p->add_to_input(input_p);
    
    EXPECT_DOUBLE_EQ(1, layer_p->input()[0][0])
        << "add_to_input not working.";
    EXPECT_DOUBLE_EQ(2, layer_p->input()[0][1])
        << "add_to_input not working.";
    EXPECT_DOUBLE_EQ(3, layer_p->input()[0][2])
        << "add_to_input not working.";
    EXPECT_DOUBLE_EQ(4, layer_p->input()[0][3])
        << "add_to_input not working.";
    
    layer_p->forward();
    
    EXPECT_DOUBLE_EQ(2.1931757358900148, layer_p->output()[0][0])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(3.5231883119115293, layer_p->output()[0][1])
        << "Forward pass incorrect.";
    
    double* outerror_p = new double[2];
    outerror_p[0] = -1;
    outerror_p[1] = 1;
    
    layer_p->add_to_outerror(outerror_p);
    layer_p->backward();
    
    EXPECT_DOUBLE_EQ(-0.58983579972444555, layer_p->inerror()[0][0])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.41997434161402647, layer_p->inerror()[0][1])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(-0.7310585786300049, layer_p->inerror()[0][2])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.88079707797788231, layer_p->inerror()[0][3])
        << "Backward pass incorrect.";
}


TEST(TestModules, PartialSoftmaxLayer) {
    PartialSoftmaxLayer* layer_p = new PartialSoftmaxLayer(4, 2);
    
    double* input_p = new double[5];
    input_p[0] = 2;
    input_p[1] = 4;
    input_p[2] = 4;
    input_p[3] = 8;
    
    layer_p->add_to_input(input_p);
    
    ASSERT_DOUBLE_EQ(2, layer_p->input()[0][0])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(4, layer_p->input()[0][1])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(4, layer_p->input()[0][2])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(8, layer_p->input()[0][3])
        << "add_to_input not working.";
    
    layer_p->forward();
    
    EXPECT_DOUBLE_EQ(0.11920292202211756, layer_p->output()[0][0])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.88079707797788243, layer_p->output()[0][1])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.098446325560013689, layer_p->output()[0][2])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.90155367443998624, layer_p->output()[0][3])
        << "Forward pass incorrect.";
    
    double* outerror_p = new double[5];
    outerror_p[0] = 2;
    outerror_p[1] = 4;
    outerror_p[2] = 1;
    outerror_p[3] = 3;
    
    layer_p->add_to_outerror(outerror_p);
    layer_p->backward();
    
    EXPECT_DOUBLE_EQ(2, layer_p->inerror()[0][0])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(4, layer_p->inerror()[0][1])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(1, layer_p->inerror()[0][2])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(3, layer_p->inerror()[0][3])
        << "Backward pass incorrect.";
}


TEST(TestModules, LstmLayer) {
    // ASSERT_TRUE(false) << "Test ist inactive at the moment.";
    LstmLayer* layer_p = new LstmLayer(1);
    
    double* input_p = new double[4];
    input_p[0] = 1;
    input_p[1] = 2;
    input_p[2] = 3;
    input_p[3] = 4;
    
    layer_p->add_to_input(input_p);
    
    ASSERT_DOUBLE_EQ(1, layer_p->input()[0][0])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(2, layer_p->input()[0][1])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(3, layer_p->input()[0][2])
        << "add_to_input not working.";
    ASSERT_DOUBLE_EQ(4, layer_p->input()[0][3])
        << "add_to_input not working.";
    
    layer_p->forward();
    
    EXPECT_DOUBLE_EQ(0.61032029727785686, layer_p->output()[0][0])
        << "Forward pass incorrect.";
        
    EXPECT_DOUBLE_EQ(0.72744331388925076, layer_p->state()[0][0])
        << "State incorrect.";
        
    input_p[0] = 1;
    input_p[1] = -2;
    input_p[2] = 3;
    input_p[3] = 4;
    
    layer_p->add_to_input(input_p);
    
    layer_p->forward();
    
    EXPECT_DOUBLE_EQ(0.65979239214347674, layer_p->output()[1][0])
        << "Forward pass incorrect.";
        
    EXPECT_DOUBLE_EQ(0.81415668251030193, layer_p->state()[1][0])
        << "State incorrect.";
    
    double* outerror_p = new double[1];
    outerror_p[0] = -1;
    
    layer_p->add_to_outerror(outerror_p);
    
    ASSERT_DOUBLE_EQ(0, layer_p->outerror()[0][0])
        << "add_to_error does not work.";
    ASSERT_DOUBLE_EQ(-1, layer_p->outerror()[1][0])
        << "add_to_error does not work.";
    
    layer_p->backward();
    
    EXPECT_DOUBLE_EQ(-0.10539391322654772, layer_p->inerror()[1][0])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(-0.041145334820469018, layer_p->inerror()[1][1])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(-0.0038855598463623424, layer_p->inerror()[1][2])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(-0.011867164496483215, layer_p->inerror()[1][3])
        << "Backward pass incorrect.";
        
    outerror_p[0] = 2;
    
    layer_p->add_to_outerror(outerror_p);
    
    ASSERT_DOUBLE_EQ(2, layer_p->outerror()[0][0])
        << "add_to_error does not work.";
    
    layer_p->backward();
    
    EXPECT_DOUBLE_EQ(0.22326096032509266, layer_p->inerror()[0][0])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0, layer_p->inerror()[0][1])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.0082309670088325619, layer_p->inerror()[0][2])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.021954698021931326, layer_p->inerror()[0][3])
        << "Backward pass incorrect.";
}


TEST(TestModules, MdlstmLayer) {
    MdlstmLayer* layer_p = new MdlstmLayer(2, 1);
    
    double* input_p = new double[10];
    input_p[0] = -2;
    input_p[1] = 1;
    input_p[2] = 2;
    input_p[3] = 3;
    input_p[4] = 4;
    input_p[5] = 5;
    input_p[6] = 6;
    input_p[7] = 7;
    input_p[8] = 8;
    input_p[9] = 9;
    
    layer_p->add_to_input(input_p);
    layer_p->forward();
    
    EXPECT_DOUBLE_EQ(0.99752618538000837, layer_p->output()[0][0])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.99908893224240791, layer_p->output()[0][1])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(7.1654995964142723, layer_p->output()[0][2])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(9.3041593430291716, layer_p->output()[0][3])
        << "Forward pass incorrect.";
    
    double* outerror_p = new double[5];
    outerror_p[0] = -1;
    outerror_p[1] = 3;
    outerror_p[2] = 1;
    outerror_p[3] = 2;
    
    layer_p->add_to_outerror(outerror_p);
    layer_p->backward();
    
    EXPECT_DOUBLE_EQ(0.10492291615431382, layer_p->inerror()[0][0])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.39318818296939362, layer_p->inerror()[0][1])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.83994668169309272, layer_p->inerror()[0][2])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.81317991556297775, layer_p->inerror()[0][3])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.0001598448588049732, layer_p->inerror()[0][4])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.000265495970626106, layer_p->inerror()[0][5])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(-0.0024665063453200445, layer_p->inerror()[0][6])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.0027306634950956059, layer_p->inerror()[0][7])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0.8807949791042492, layer_p->inerror()[0][8])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(1.9051483483108722, layer_p->inerror()[0][9])
        << "Backward pass incorrect.";
}


TEST(TestConnections, IdentityConnection) {
    LinearLayer* inlayer_p = new LinearLayer(2);
    LinearLayer* outlayer_p = new LinearLayer(2);
    
    inlayer_p->input()[0][0] = 2.;
    inlayer_p->input()[0][1] = 3.;
    
    IdentityConnection* con_p = new IdentityConnection(inlayer_p, outlayer_p);
    
    inlayer_p->forward();
    con_p->forward();
    outlayer_p->forward();
    
    EXPECT_DOUBLE_EQ(2., outlayer_p->output()[0][0])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(3., outlayer_p->output()[0][1])
        << "Forward pass incorrect.";
    
    outlayer_p->outerror()[0][0] = 0.5;
    outlayer_p->outerror()[0][1] = 1.2;
    outlayer_p->backward();
    
    con_p->backward();
    inlayer_p->backward();
    
    EXPECT_DOUBLE_EQ(0.5, inlayer_p->outerror()[0][0])
            << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(1.2, inlayer_p->outerror()[0][1])
            << "Backward pass incorrect.";
}


TEST(TestConnections, RecurrentIdentityConnection) {
    LinearLayer* inlayer_p = new LinearLayer(2);
    LinearLayer* outlayer_p = new LinearLayer(2);
    IdentityConnection* con_p = new IdentityConnection(inlayer_p, outlayer_p);
    
    inlayer_p->set_mode(Component::Sequential);
    outlayer_p->set_mode(Component::Sequential);
    con_p->set_mode(Component::Sequential);
    con_p->set_recurrent(1);
    
    // First some data that should not have any effect immediately.
    
    inlayer_p->input()[0][0] = 2.;
    inlayer_p->input()[0][1] = 3.;
    
    inlayer_p->forward();
    con_p->forward();
    outlayer_p->forward();
    
    EXPECT_DOUBLE_EQ(0, outlayer_p->output()[0][0])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(0, outlayer_p->output()[0][1])
        << "Forward pass incorrect.";
    
    // Now the previous information should pass through.
            
    inlayer_p->input()[0][0] = 0.5;
    inlayer_p->input()[0][1] = 3.2;
    
    inlayer_p->forward();
    con_p->forward();
    outlayer_p->forward();
    
    EXPECT_DOUBLE_EQ(2, outlayer_p->output()[1][0])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(3, outlayer_p->output()[1][1])
        << "Forward pass incorrect.";
    
    // Let's do a backward pass. 
    
    outlayer_p->outerror()[1][0] = 0.5;
    outlayer_p->outerror()[1][1] = 1.2;

    outlayer_p->backward();
    con_p->backward();
    inlayer_p->backward();
    
    EXPECT_DOUBLE_EQ(0, inlayer_p->outerror()[1][0])
            << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0, inlayer_p->outerror()[1][1])
            << "Backward pass incorrect.";
            
    outlayer_p->outerror()[0][0] = 2.3;
    outlayer_p->outerror()[0][1] = 1.8;

    outlayer_p->backward();
    con_p->backward();
    inlayer_p->backward();
    
    EXPECT_DOUBLE_EQ(0.5, inlayer_p->outerror()[0][0])
            << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(1.2, inlayer_p->outerror()[0][1])
            << "Backward pass incorrect.";
}


TEST(TestConnections, IdentityConnectionSliced) {
    LinearLayer* inlayer_p = new LinearLayer(2);
    LinearLayer* outlayer_p = new LinearLayer(2);
    
    inlayer_p->input()[0][0] = 2.;
    inlayer_p->input()[0][1] = 3.;
    
    IdentityConnection* con_p = \
        new IdentityConnection(inlayer_p, outlayer_p, 0, 1, 1, 2);
    
    ASSERT_EQ(0, con_p->get_incomingstart())
        << "_incomingstart not initialized properly.";
    ASSERT_EQ(1, con_p->get_incomingstop())
        << "_incomingstop not initialized properly.";
    ASSERT_EQ(1, con_p->get_outgoingstart())
        << "_outgoingstart not initialized properly.";
    ASSERT_EQ(2, con_p->get_outgoingstop())
        << "_ougoingstop not initialized properly.";
    
    inlayer_p->forward();
    con_p->forward();
    outlayer_p->forward();
    
    ASSERT_DOUBLE_EQ(0., outlayer_p->output()[0][0])
        << "Forward pass incorrect.";
    ASSERT_DOUBLE_EQ(2., outlayer_p->output()[0][1])
            << "Forward pass incorrect.";
    
    outlayer_p->outerror()[0][0] = 0.5;
    outlayer_p->outerror()[0][1] = 1.2;
    outlayer_p->backward();
    
    con_p->backward();
    inlayer_p->backward();
    
    EXPECT_DOUBLE_EQ(1.2, inlayer_p->outerror()[0][0])
            << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0, inlayer_p->outerror()[0][1])
            << "Backward pass incorrect.";
}


TEST(TestConnections, FullConnection) {
    LinearLayer* inlayer_p = new LinearLayer(2);
    LinearLayer* outlayer_p = new LinearLayer(3);
    
    inlayer_p->input()[0][0] = 2.;
    inlayer_p->input()[0][1] = 3.;
    
    FullConnection* con_p = new FullConnection(inlayer_p, outlayer_p);
    con_p->get_parameters()[0] = 0; 
    con_p->get_parameters()[1] = 1; 
    con_p->get_parameters()[2] = 2; 
    con_p->get_parameters()[3] = 3; 
    con_p->get_parameters()[4] = 4; 
    con_p->get_parameters()[5] = 5;
    
    inlayer_p->forward();
    con_p->forward();
    
    EXPECT_DOUBLE_EQ(3, outlayer_p->input()[0][0])
        << "Forward pass not working.";
    EXPECT_DOUBLE_EQ(13, outlayer_p->input()[0][1])
        << "Forward pass not working.";
    EXPECT_DOUBLE_EQ(23, outlayer_p->input()[0][2])
        << "Forward pass not working.";
    
    outlayer_p->forward();
    outlayer_p->outerror()[0][0] = 0.5;
    outlayer_p->outerror()[0][1] = 1.2;
    outlayer_p->outerror()[0][2] = 3.4;
    outlayer_p->backward();
    con_p->backward();
    
    EXPECT_DOUBLE_EQ(16, inlayer_p->outerror()[0][0])
            << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(21.1, inlayer_p->outerror()[0][1])
            << "Backward pass incorrect.";

    EXPECT_DOUBLE_EQ(1, con_p->get_derivatives()[0])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(1.5, con_p->get_derivatives()[1])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(2.4, con_p->get_derivatives()[2])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(3.6, con_p->get_derivatives()[3])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(6.8, con_p->get_derivatives()[4])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(10.2, con_p->get_derivatives()[5])
        << "Backward pass not working.";
}


TEST(TestConnections, LinearConnection) {
    LinearLayer* inlayer_p = new LinearLayer(2);
    LinearLayer* outlayer_p = new LinearLayer(2);
    
    inlayer_p->input()[0][0] = 2.;
    inlayer_p->input()[0][1] = 3.;
    
    LinearConnection* con_p = new LinearConnection(inlayer_p, outlayer_p);
    con_p->get_parameters()[0] = 1.5; 
    con_p->get_parameters()[1] = 2; 
    
    inlayer_p->forward();
    con_p->forward();
    
    EXPECT_DOUBLE_EQ(3, outlayer_p->input()[0][0])
        << "Forward pass not working.";
    EXPECT_DOUBLE_EQ(6, outlayer_p->input()[0][1])
        << "Forward pass not working.";
    
    outlayer_p->forward();
    outlayer_p->outerror()[0][0] = 0.5;
    outlayer_p->outerror()[0][1] = 1.2;
    outlayer_p->backward();
    con_p->backward();
    
    EXPECT_DOUBLE_EQ(0.75, inlayer_p->outerror()[0][0])
            << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(2.4, inlayer_p->outerror()[0][1])
            << "Backward pass incorrect.";

    EXPECT_DOUBLE_EQ(1, con_p->get_derivatives()[0])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(3.6, con_p->get_derivatives()[1])
        << "Backward pass not working.";
}


TEST(TestConnections, FullConnectionSliced) {
    LinearLayer* inlayer_p = new LinearLayer(2);
    LinearLayer* outlayer_p = new LinearLayer(3);
    
    inlayer_p->input()[0][0] = 2.;
    inlayer_p->input()[0][1] = 3.;
    
    FullConnection* con_p = new FullConnection(inlayer_p, outlayer_p, 0, 1, 0, 4);
    
    ASSERT_EQ(0, con_p->get_incomingstart())
        << "_incomingstart not initialized properly.";
    ASSERT_EQ(1, con_p->get_incomingstop())
        << "_incomingstop not initialized properly.";
    ASSERT_EQ(0, con_p->get_outgoingstart())
        << "_outgoingstart not initialized properly.";
    ASSERT_EQ(4, con_p->get_outgoingstop())
        << "_ougoingstop not initialized properly.";
    
    con_p->get_parameters()[0] = -1;
    con_p->get_parameters()[1] = 1; 
    con_p->get_parameters()[2] = 2; 
    
    inlayer_p->forward();
    con_p->forward();
    
    ASSERT_DOUBLE_EQ(-2, outlayer_p->input()[0][0])
        << "Forward pass not working.";
    ASSERT_DOUBLE_EQ(2, outlayer_p->input()[0][1])
        << "Forward pass not working.";
    ASSERT_DOUBLE_EQ(4, outlayer_p->input()[0][2])
        << "Forward pass not working.";
    
    outlayer_p->forward();
    outlayer_p->outerror()[0][0] = 0.5;
    outlayer_p->outerror()[0][1] = 1.2;
    outlayer_p->outerror()[0][2] = 3.4;
    outlayer_p->backward();
    con_p->backward();
    
    EXPECT_DOUBLE_EQ(1, con_p->get_derivatives()[0])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(2.4, con_p->get_derivatives()[1])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(6.8, con_p->get_derivatives()[2])
        << "Backward pass not working.";
}


TEST(TestConnections, RecurrentFullConnection) {
    LinearLayer* inlayer_p = new LinearLayer(2);
    LinearLayer* outlayer_p = new LinearLayer(3);
    FullConnection* con_p = new FullConnection(inlayer_p, outlayer_p);
    
    inlayer_p->set_mode(Component::Sequential);
    outlayer_p->set_mode(Component::Sequential);
    con_p->set_mode(Component::Sequential);
    con_p->set_recurrent(1);
    
    con_p->get_parameters()[0] = 0; 
    con_p->get_parameters()[1] = 1; 
    con_p->get_parameters()[2] = 2; 
    con_p->get_parameters()[3] = 3; 
    con_p->get_parameters()[4] = 4; 
    con_p->get_parameters()[5] = 5;

    EXPECT_DOUBLE_EQ(0, outlayer_p->input()[0][0])
        << "Buffer not properly initialized.";
    EXPECT_DOUBLE_EQ(0, outlayer_p->input()[0][1])
        << "Buffer not properly initialized.";
    EXPECT_DOUBLE_EQ(0, outlayer_p->input()[0][2])
        << "Buffer not properly initialized.";

    // First some data that should not have any effect immediately.
    
    inlayer_p->input()[0][0] = 2.;
    inlayer_p->input()[0][1] = 3.;
    
    inlayer_p->forward();
    con_p->forward();
    outlayer_p->forward();
    
    EXPECT_DOUBLE_EQ(0, outlayer_p->input()[0][0])
        << "Forward pass not working.";
    EXPECT_DOUBLE_EQ(0, outlayer_p->input()[0][1])
        << "Forward pass not working.";
    EXPECT_DOUBLE_EQ(0, outlayer_p->input()[0][2])
        << "Forward pass not working.";
    
    // Now the previous information should pass through.
    inlayer_p->input()[1][0] = 0.5;
    inlayer_p->input()[1][1] = 3.2;
    
    inlayer_p->forward();
    con_p->forward();
    outlayer_p->forward();
    
    EXPECT_DOUBLE_EQ(3, outlayer_p->output()[1][0])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(13, outlayer_p->output()[1][1])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(23, outlayer_p->output()[1][2])
        << "Forward pass incorrect.";
    
    // Let's do a backward pass. 
    
    outlayer_p->outerror()[1][0] = 0.5;
    outlayer_p->outerror()[1][1] = 1.2;
    outlayer_p->outerror()[1][2] = 3.4;

    outlayer_p->backward();
    con_p->backward();
    inlayer_p->backward();
    
    EXPECT_DOUBLE_EQ(0, inlayer_p->outerror()[1][0])
            << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0, inlayer_p->outerror()[1][1])
            << "Backward pass incorrect.";
            
    EXPECT_DOUBLE_EQ(0, con_p->get_derivatives()[0])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(0, con_p->get_derivatives()[1])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(0, con_p->get_derivatives()[2])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(0, con_p->get_derivatives()[3])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(0, con_p->get_derivatives()[4])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(0, con_p->get_derivatives()[5])
        << "Backward pass not working.";

    outlayer_p->outerror()[0][0] = 2.3;
    outlayer_p->outerror()[0][1] = 1.8;
    outlayer_p->outerror()[0][1] = 1.2;

    outlayer_p->backward();
    con_p->backward();
    inlayer_p->backward();
    
    EXPECT_DOUBLE_EQ(1, con_p->get_derivatives()[0])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(1.5, con_p->get_derivatives()[1])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(2.4, con_p->get_derivatives()[2])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(3.6, con_p->get_derivatives()[3])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(6.8, con_p->get_derivatives()[4])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(10.2, con_p->get_derivatives()[5])
        << "Backward pass not working.";
}


TEST(TestConnections, DeepRecurrentFullConnection) {
    LinearLayer* inlayer_p = new LinearLayer(2);
    LinearLayer* outlayer_p = new LinearLayer(3);
    FullConnection* con_p = new FullConnection(inlayer_p, outlayer_p);
    
    inlayer_p->set_mode(Component::Sequential);
    outlayer_p->set_mode(Component::Sequential);
    con_p->set_mode(Component::Sequential);
    con_p->set_recurrent(2);
    
    con_p->get_parameters()[0] = 0; 
    con_p->get_parameters()[1] = 1; 
    con_p->get_parameters()[2] = 2; 
    con_p->get_parameters()[3] = 3; 
    con_p->get_parameters()[4] = 4; 
    con_p->get_parameters()[5] = 5;

    EXPECT_DOUBLE_EQ(0, outlayer_p->input()[0][0])
        << "Buffer not properly initialized.";
    EXPECT_DOUBLE_EQ(0, outlayer_p->input()[0][1])
        << "Buffer not properly initialized.";
    EXPECT_DOUBLE_EQ(0, outlayer_p->input()[0][2])
        << "Buffer not properly initialized.";

    // FIRST STEP FORWARD
    inlayer_p->input()[0][0] = 2.;
    inlayer_p->input()[0][1] = 3.;
    
    inlayer_p->forward();
    con_p->forward();
    outlayer_p->forward();

    EXPECT_DOUBLE_EQ(0, outlayer_p->input()[0][0])
        << "Forward pass not working.";
    EXPECT_DOUBLE_EQ(0, outlayer_p->input()[0][1])
        << "Forward pass not working.";
    EXPECT_DOUBLE_EQ(0, outlayer_p->input()[0][2])
        << "Forward pass not working.";


    // SECOND STEP FORWARD

    inlayer_p->input()[1][0] = 0.;
    inlayer_p->input()[1][1] = 0.;
    
    inlayer_p->forward();
    con_p->forward();
    outlayer_p->forward();

    EXPECT_DOUBLE_EQ(0, outlayer_p->input()[0][0])
        << "Forward pass not working.";
    EXPECT_DOUBLE_EQ(0, outlayer_p->input()[0][1])
        << "Forward pass not working.";
    EXPECT_DOUBLE_EQ(0, outlayer_p->input()[0][2])
        << "Forward pass not working.";

    // THIRD STEP FORWARD
    
    inlayer_p->input()[2][0] = 0;
    inlayer_p->input()[2][1] = 0;
    
    inlayer_p->forward();
    con_p->forward();
    outlayer_p->forward();
    
    EXPECT_DOUBLE_EQ(3, outlayer_p->output()[2][0])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(13, outlayer_p->output()[2][1])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(23, outlayer_p->output()[2][2])
        << "Forward pass incorrect.";
    
    
    // FIRST STEP BACKWARD
    
    outlayer_p->outerror()[2][0] = 0.5;
    outlayer_p->outerror()[2][1] = 1.2;
    outlayer_p->outerror()[2][2] = 3.4;

    outlayer_p->backward();
    con_p->backward();
    inlayer_p->backward();

    EXPECT_DOUBLE_EQ(0.5, outlayer_p->inerror()[2][0])
            << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(1.2, outlayer_p->inerror()[2][1])
            << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(3.4, outlayer_p->inerror()[2][2])
            << "Backward pass incorrect.";
    
    EXPECT_DOUBLE_EQ(0, inlayer_p->outerror()[2][0])
            << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0, inlayer_p->outerror()[2][1])
            << "Backward pass incorrect.";
            
    EXPECT_DOUBLE_EQ(0, con_p->get_derivatives()[0])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(0, con_p->get_derivatives()[1])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(0, con_p->get_derivatives()[2])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(0, con_p->get_derivatives()[3])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(0, con_p->get_derivatives()[4])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(0, con_p->get_derivatives()[5])
        << "Backward pass not working.";

    
    // SECOND STEP BACKWARDS
    
    outlayer_p->outerror()[1][0] = -1;
    outlayer_p->outerror()[1][1] = -1;
    outlayer_p->outerror()[1][2] = -1;

    outlayer_p->backward();
    con_p->backward();
    inlayer_p->backward();
    
    EXPECT_DOUBLE_EQ(0, inlayer_p->outerror()[1][0])
            << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0, inlayer_p->outerror()[1][1])
            << "Backward pass incorrect.";
            
    EXPECT_DOUBLE_EQ(0, con_p->get_derivatives()[0])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(0, con_p->get_derivatives()[1])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(0, con_p->get_derivatives()[2])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(0, con_p->get_derivatives()[3])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(0, con_p->get_derivatives()[4])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(0, con_p->get_derivatives()[5])
        << "Backward pass not working.";


    // THIRD STEP BACKWARDS

    outlayer_p->outerror()[0][0] = -1;
    outlayer_p->outerror()[0][1] = -1;
    outlayer_p->outerror()[0][1] = -1;

    outlayer_p->backward();
    con_p->backward();
    inlayer_p->backward();
    
    EXPECT_DOUBLE_EQ(0, inlayer_p->outerror()[2][0])
            << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0, inlayer_p->outerror()[2][1])
            << "Backward pass incorrect.";

    EXPECT_DOUBLE_EQ(0, inlayer_p->outerror()[1][0])
            << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(0, inlayer_p->outerror()[1][1])
            << "Backward pass incorrect.";
    
    EXPECT_DOUBLE_EQ(16, inlayer_p->outerror()[0][0])
            << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(21.1, inlayer_p->outerror()[0][1])
            << "Backward pass incorrect.";
    
    EXPECT_DOUBLE_EQ(1, con_p->get_derivatives()[0])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(1.5, con_p->get_derivatives()[1])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(2.4, con_p->get_derivatives()[2])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(3.6, con_p->get_derivatives()[3])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(6.8, con_p->get_derivatives()[4])
        << "Backward pass not working.";
    EXPECT_DOUBLE_EQ(10.2, con_p->get_derivatives()[5])
        << "Backward pass not working.";
}


TEST(TestNetwork, TestCopyResult) {
    Network* net_p = new Network();
    
    LinearLayer* inlayer_p = new LinearLayer(2);
    LinearLayer* outlayer_p = new LinearLayer(2);
    IdentityConnection* con_p = new IdentityConnection(inlayer_p, outlayer_p);

    net_p->add_module(inlayer_p, Network::InputModule);
    net_p->add_module(outlayer_p, Network::OutputModule);
    net_p->add_connection(con_p);
    
    double* input_p = new double[2];
    input_p[0] = 2;
    input_p[1] = 4;

    double* output_p = new double[2];
    output_p[0] = 0;
    output_p[1] = 0;
    
    net_p->activate(input_p, output_p);
    
    ASSERT_DOUBLE_EQ(2, output_p[0])
        << "Data not copied correctly into inmodule.";
    ASSERT_DOUBLE_EQ(4, output_p[1])
        << "Data not copied correctly into inmodule.";
}


TEST(TestNetwork, TestTwoLayerNetwork) {
    Network* net_p = new Network();
    
    LinearLayer* inlayer_p = new LinearLayer(2);
    LinearLayer* outlayer_p = new LinearLayer(2);
    FullConnection* con_p = new FullConnection(inlayer_p, outlayer_p);
    
    con_p->get_parameters()[0] = 0.5;
    con_p->get_parameters()[1] = -2;
    con_p->get_parameters()[2] = 1.2;
    con_p->get_parameters()[3] = 4;
    
    net_p->add_module(inlayer_p, Network::InputModule);
    net_p->add_module(outlayer_p, Network::OutputModule);
    net_p->add_connection(con_p);
    
    double* input_p = new double[2];
    input_p[0] = 2;
    input_p[1] = 4;
    
    const double* output_p = net_p->activate(input_p);

    ASSERT_DOUBLE_EQ(2, net_p->input()[0][0])
        << "Data not copied correctly into network.";
    ASSERT_DOUBLE_EQ(4, net_p->input()[0][1])
        << "Data not copied correctly into network.";

    ASSERT_DOUBLE_EQ(2, inlayer_p->input()[0][0])
        << "Data not copied correctly into inmodule.";
    ASSERT_DOUBLE_EQ(4, inlayer_p->input()[0][1])
        << "Data not copied correctly into inmodule.";
    
    EXPECT_DOUBLE_EQ(-7, outlayer_p->output()[0][0])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(18.4, outlayer_p->output()[0][1])
        << "Forward pass incorrect.";
        
    EXPECT_DOUBLE_EQ(-7, output_p[0])
        << "activate() Result not correct.";
    EXPECT_DOUBLE_EQ(18.4, output_p[1])
        << "activate() Result not correct.";

        
    double* outerror_p = new double[2];
    outerror_p[0] = 3;
    outerror_p[1] = -2;
    const double* inerror_p = net_p->back_activate(outerror_p);
    
    ASSERT_DOUBLE_EQ(3, net_p->outerror()[0][0])
        << "Error not copied correctly into network outerror buffer.";
    ASSERT_DOUBLE_EQ(-2, net_p->outerror()[0][1])
        << "Error not copied correctly into network outerror buffer.";

    ASSERT_DOUBLE_EQ(3, outlayer_p->outerror()[0][0])
        << "Error not copied correctly into outlayer outerror buffer.";
    ASSERT_DOUBLE_EQ(-2, outlayer_p->outerror()[0][1])
        << "Error not copied correctly into outlayer outerror buffer.";

    EXPECT_DOUBLE_EQ(-0.9, inlayer_p->inerror()[0][0])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(-14, inlayer_p->inerror()[0][1])
        << "Backward pass incorrect.";

    EXPECT_DOUBLE_EQ(-0.9, net_p->inerror()[0][0])
        << "Error not copied into network inerror buffer.";
    EXPECT_DOUBLE_EQ(-14, net_p->inerror()[0][1])
        << "Error not copied into network inerror buffer.";

    EXPECT_DOUBLE_EQ(-0.9, inerror_p[0])
        << "Error not returned correctly.";
    EXPECT_DOUBLE_EQ(-14, inerror_p[1])
        << "Error not returned correctly.";

    EXPECT_DOUBLE_EQ(6, con_p->get_derivatives()[0])
        << "Derivatives incorrect.";
    EXPECT_DOUBLE_EQ(12, con_p->get_derivatives()[1])
        << "Derivatives incorrect.";
    EXPECT_DOUBLE_EQ(-4, con_p->get_derivatives()[2])
        << "Derivatives incorrect.";
    EXPECT_DOUBLE_EQ(-8, con_p->get_derivatives()[3])
        << "Derivatives incorrect.";

    net_p->clear();
    
    for (int i = 0; i < net_p->input().size(); i++)
    {
        for (int j = 0; j < net_p->input().rowsize(); j++)
        {
            ASSERT_DOUBLE_EQ(0, net_p->input()[i][j])
                << "Buffer has not been set to zero at " << i << " " << j;
            
        }
    }
}
 
        
TEST(TestNetwork, TestRecurrentLayerNetwork) {
    Network* net_p = new Network();
    net_p->set_mode(Component::Sequential);
    
    LinearLayer* inlayer_p = new LinearLayer(2);
    inlayer_p->set_mode(Component::Sequential);

    LinearLayer* outlayer_p = new LinearLayer(2);
    outlayer_p->set_mode(Component::Sequential);

    IdentityConnection* con_p = new IdentityConnection(inlayer_p, outlayer_p);
    con_p->set_mode(Component::Sequential);

    IdentityConnection* rcon_p = new IdentityConnection(inlayer_p, outlayer_p);
    rcon_p->set_mode(Component::Sequential);
    rcon_p->set_recurrent(1);
    
    net_p->add_module(inlayer_p, Network::InputModule);
    net_p->add_module(outlayer_p, Network::OutputModule);
    net_p->add_connection(con_p);
    net_p->add_connection(rcon_p);
    
    double* input_p = new double[2];
    input_p[0] = 2;
    input_p[1] = 4;
    
    net_p->activate(input_p);

    EXPECT_DOUBLE_EQ(2, outlayer_p->output()[0][0])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(4, outlayer_p->output()[0][1])
        << "Forward pass incorrect.";

    input_p[0] = 3;
    input_p[1] = 6;
    net_p->activate(input_p);

    EXPECT_DOUBLE_EQ(5, outlayer_p->output()[1][0])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(10, outlayer_p->output()[1][1])
        << "Forward pass incorrect.";

    double* outerror_p = new double[2];
    outerror_p[0] = 3;
    outerror_p[1] = -2;
    net_p->back_activate(outerror_p);

    EXPECT_DOUBLE_EQ(3, net_p->inerror()[1][0])
        << "Error not copied into network inerror buffer.";
    EXPECT_DOUBLE_EQ(-2, net_p->inerror()[1][1])
        << "Error not copied into network inerror buffer.";

    outerror_p[0] = 6;
    outerror_p[1] = -3;
    net_p->back_activate(outerror_p);

    EXPECT_DOUBLE_EQ(9, net_p->inerror()[0][0])
        << "Error not returned correctly.";
    EXPECT_DOUBLE_EQ(-5, net_p->inerror()[0][1])
        << "Error not returned correctly.";
}
        
        
TEST(TestNetwork, TestRecurrentNetworkTimesteps) {
    Network* net_p = new Network();
    
    LinearLayer* inlayer_p = new LinearLayer(2);
    LinearLayer* outlayer_p = new LinearLayer(2);
    FullConnection* con_p = new FullConnection(inlayer_p, outlayer_p);
    
    net_p->add_module(inlayer_p, Network::InputModule);
    net_p->add_module(outlayer_p, Network::OutputModule);
    net_p->add_connection(con_p);
    
    net_p->set_mode(Component::Sequential);
    inlayer_p->set_mode(Component::Sequential);
    outlayer_p->set_mode(Component::Sequential);
    con_p->set_mode(Component::Sequential);
    
    double* input_p = new double[3];
    input_p[0] = 2;
    input_p[1] = 4;
    input_p[1] = 6;

    ASSERT_EQ(0, net_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(0, inlayer_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(0, outlayer_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(0, con_p->timestep())
        << "Wrong timestep.";

    net_p->activate(input_p);
    ASSERT_EQ(1, net_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(1, inlayer_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(1, outlayer_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(1, con_p->timestep())
        << "Wrong timestep.";

    net_p->activate(input_p);
    ASSERT_EQ(2, net_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(2, inlayer_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(2, outlayer_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(2, con_p->timestep())
        << "Wrong timestep.";

    net_p->activate(input_p);
    ASSERT_EQ(3, net_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(3, inlayer_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(3, outlayer_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(3, con_p->timestep())
        << "Wrong timestep.";

    net_p->back_activate(input_p);
    ASSERT_EQ(2, net_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(2, inlayer_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(2, outlayer_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(2, con_p->timestep())
        << "Wrong timestep.";

    net_p->back_activate(input_p);
    ASSERT_EQ(1, net_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(1, net_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(1, inlayer_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(1, outlayer_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(1, con_p->timestep())
        << "Wrong timestep.";

    net_p->back_activate(input_p);
    ASSERT_EQ(0, net_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(0, inlayer_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(0, outlayer_p->timestep())
        << "Wrong timestep.";
    ASSERT_EQ(0, con_p->timestep())
        << "Wrong timestep.";
}
        
        
TEST(TestNetwork, TestMdrnn)
{
    Mdrnn<LinearLayer> net(2, 1);
    net.set_sequence_shape(0, 2);
    net.set_sequence_shape(1, 2);
    
    double* params_p = new double[2];
    params_p[0] = 0.5;
    params_p[1] = 2;
    
    net.set_parameters(params_p);
    
    double* input_p = new double[4];
    input_p[0] = 1;
    input_p[1] = 2;
    input_p[2] = 3;
    input_p[3] = 4;
    
    const double* output_p = net.activate(input_p);

    ASSERT_DOUBLE_EQ(1, net.input()[0][0]) 
        << "Networks' input not filled correctly.";
    ASSERT_DOUBLE_EQ(2, net.input()[0][1])
        << "Networks' input not filled correctly.";
    ASSERT_DOUBLE_EQ(3, net.input()[0][2])
        << "Networks' input not filled correctly.";
    ASSERT_DOUBLE_EQ(4, net.input()[0][3])
        << "Networks' input not filled correctly.";
    
    EXPECT_DOUBLE_EQ(1, output_p[0]) 
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(2.5, output_p[1])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(5, output_p[2])
        << "Forward pass incorrect.";
    EXPECT_DOUBLE_EQ(11.5, output_p[3])
        << "Forward pass incorrect.";
        
    double* outerror_p = new double[4];
    
    outerror_p[0] = 2;
    outerror_p[1] = 4;
    outerror_p[2] = 8;
    outerror_p[3] = 10;
    
    const double* inerror_p = net.back_activate(outerror_p);

    ASSERT_DOUBLE_EQ(2, net.outerror()[0][0]) 
        << "Networks' outerror not filled correctly.";
    ASSERT_DOUBLE_EQ(4, net.outerror()[0][1])
        << "Networks' outerror not filled correctly.";
    ASSERT_DOUBLE_EQ(8, net.outerror()[0][2])
        << "Networks' outerror not filled correctly.";
    ASSERT_DOUBLE_EQ(10, net.outerror()[0][3])
        << "Networks' outerror not filled correctly.";

    EXPECT_DOUBLE_EQ(40, net.inerror()[0][0]) 
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(24, net.inerror()[0][1])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(13, net.inerror()[0][2])
        << "Backward pass incorrect.";
    EXPECT_DOUBLE_EQ(10, net.inerror()[0][3])
        << "Backward pass incorrect.";
    
    EXPECT_DOUBLE_EQ(net.inerror()[0][0], inerror_p[0]) 
        << "back_activate copy not correct.";
    EXPECT_DOUBLE_EQ(net.inerror()[0][1], inerror_p[1])
        << "back_activate copy not correct.";
    EXPECT_DOUBLE_EQ(net.inerror()[0][2], inerror_p[2])
        << "back_activate copy not correct.";
    EXPECT_DOUBLE_EQ(net.inerror()[0][3], inerror_p[3])
        << "back_activate copy not correct.";
}


TEST(TestNetwork, NetworkClearConnection)
{
    Network* net_p = new Network();
    LinearLayer* input = new LinearLayer(1);
    LinearLayer* output = new LinearLayer(1);
    FullConnection* con = \
        new FullConnection(input, output);
        
    double* input_p = new double[1];
        
    net_p->add_module(input, Network::InputModule);
    net_p->add_module(output, Network::OutputModule);
    net_p->add_connection(con);
    net_p->activate(input_p);
    net_p->clear();
    EXPECT_EQ(0, net_p->timestep())
        << "Timestep was not reseted.";
    EXPECT_EQ(0, input->timestep())
        << "Timestep was not reseted.";
    EXPECT_EQ(0, output->timestep())
        << "Timestep was not reseted.";
    EXPECT_EQ(0, con->timestep())
        << "Timestep was not reseted.";
}

        
}  // namespace


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
