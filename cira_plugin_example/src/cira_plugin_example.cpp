#include <cira_plugin_example/cira_plugin_example.hpp>
#include <cira_lib_bernoulli/CiraBernoulliWidget.h>
#include <models/PPOCRv4/PPOCRv4Model.hpp>
//#include <models/CiRA_PaddleOCR/CiRA_PaddleOCRModel.hpp>
cira_plugin_example::cira_plugin_example() {

}

std::shared_ptr<DataModelRegistry> cira_plugin_example::registerDataModels(std::shared_ptr<DataModelRegistry> ret) {


  /**********************
  regist model here
  example:

    ret->registerModel<Some1Model>();
    ret->registerModel<Some2Model>();

  ******************/
  ret->registerModel<PPOCRv4Model>();
  //ret->registerModel<CiRA_PaddleOCRModel>();
  setTreeWidget();
  return ret;

}

void cira_plugin_example::setTreeWidget() {

  QStringList strListNodeDataModels;
  QString category;


  /**********************
  regist model name here for drag&drop
  example:

    category = "SomeCategory";
    strListNodeDataModels << category + ",Some1"      + ",null";
    strListNodeDataModels << category + ",Some2"      + ",null";

    CiraBernoulliWidget::pluginTreeWidget->addNodeDataModels(strListNodeDataModels);

  ******************/
  category = "Customize Plugin";
  strListNodeDataModels << category + ",PPOCRv4"      + ",null";
  //strListNodeDataModels << category + ",CiRA_PPOCR"      + ",null";

  CiraBernoulliWidget::pluginTreeWidget->addNodeDataModels(strListNodeDataModels);
}
