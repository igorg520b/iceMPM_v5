#ifndef vtkHedgeHogCustom_h
#define vtkHedgeHogCustom_h

#include "vtkPolyDataAlgorithm.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"

#define VTK_USE_VECTOR 0
#define VTK_USE_NORMAL 1

class vtkHedgeHogCustom : public vtkPolyDataAlgorithm
{
public:
  static vtkHedgeHogCustom* New();
  vtkTypeMacro(vtkHedgeHogCustom, vtkPolyDataAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  vtkSetMacro(ScaleFactor, double);
  vtkGetMacro(ScaleFactor, double);

  vtkSetMacro(ArrowheadSize, double);
  vtkGetMacro(ArrowheadSize, double);

  vtkSetMacro(VectorMode, int);
  vtkGetMacro(VectorMode, int);
  void SetVectorModeToUseVector() { this->SetVectorMode(VTK_USE_VECTOR); }
  void SetVectorModeToUseNormal() { this->SetVectorMode(VTK_USE_NORMAL); }
  const char* GetVectorModeAsString();

  vtkSetMacro(OutputPointsPrecision, int);
  vtkGetMacro(OutputPointsPrecision, int);

protected:
  vtkHedgeHogCustom();
  ~vtkHedgeHogCustom() override = default;

  int RequestData(vtkInformation*, vtkInformationVector**, vtkInformationVector*) override;
  int FillInputPortInformation(int port, vtkInformation* info) override;
  double ScaleFactor;
  double ArrowheadSize;
  int VectorMode; // Orient/scale via normal or via vector data
  int OutputPointsPrecision;

private:
  vtkHedgeHogCustom(const vtkHedgeHogCustom&) = delete;
  void operator=(const vtkHedgeHogCustom&) = delete;
};

inline const char* vtkHedgeHogCustom::GetVectorModeAsString()
{
  if (this->VectorMode == VTK_USE_VECTOR)
  {
    return "UseVector";
  }
  else if (this->VectorMode == VTK_USE_NORMAL)
  {
    return "UseNormal";
  }
  else
  {
    return "Unknown";
  }
}
#endif
