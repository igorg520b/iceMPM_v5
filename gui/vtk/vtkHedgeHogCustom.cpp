#include "vtkHedgeHogCustom.h"

#include "vtkCellArray.h"
#include "vtkInformation.h"
#include "vtkInformationVector.h"
#include "vtkObjectFactory.h"
#include "vtkPointData.h"
#include "vtkCellData.h"
#include "vtkPolyData.h"
#include "vtkPointSet.h"
#include "vtkUnsignedCharArray.h"
#include "vtkNew.h"

vtkStandardNewMacro(vtkHedgeHogCustom);

vtkHedgeHogCustom::vtkHedgeHogCustom()
{
  this->ScaleFactor = 1.0;
  this->ArrowheadSize = 0.036; // Default to visually distinct length
  this->VectorMode = VTK_USE_VECTOR;
  this->OutputPointsPrecision = vtkAlgorithm::DEFAULT_PRECISION;
}

int vtkHedgeHogCustom::RequestData(vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector, vtkInformationVector* outputVector)
{
  // get the info objects
  vtkInformation* inInfo = inputVector[0]->GetInformationObject(0);
  vtkInformation* outInfo = outputVector->GetInformationObject(0);

  // get the input and output
  vtkDataSet* input = vtkDataSet::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));
  vtkPolyData* output = vtkPolyData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

  vtkIdType numPts;
  vtkPoints* newPts;
  vtkPointData* pd;
  vtkDataArray* inVectors;
  vtkDataArray* inNormals;
  vtkIdType ptId;
  int i;
  vtkIdType pts[2];
  vtkCellArray* newLines;
  double x[3], v[3];
  double newX[3];
  vtkPointData* outputPD = output->GetPointData();

  // Initialize
  //
  numPts = input->GetNumberOfPoints();
  pd = input->GetPointData();
  inVectors = pd->GetVectors();
  if (numPts < 1)
  {
    vtkErrorMacro(<< "No input data");
    return 1;
  }
  if (!inVectors && this->VectorMode == VTK_USE_VECTOR)
  {
    vtkErrorMacro(<< "No vectors in input data");
    return 1;
  }

  inNormals = pd->GetNormals();
  if (!inNormals && this->VectorMode == VTK_USE_NORMAL)
  {
    vtkErrorMacro(<< "No normals in input data");
    return 1;
  }
  outputPD->CopyAllocate(pd, 4 * numPts);

  newPts = vtkPoints::New();

  // Set the desired precision for the points in the output.
  if (this->OutputPointsPrecision == vtkAlgorithm::DEFAULT_PRECISION)
  {
    vtkPointSet* inputPointSet = vtkPointSet::SafeDownCast(input);
    if (inputPointSet)
    {
      newPts->SetDataType(inputPointSet->GetPoints()->GetDataType());
    }
    else
    {
      newPts->SetDataType(VTK_FLOAT);
    }
  }
  else if (this->OutputPointsPrecision == vtkAlgorithm::SINGLE_PRECISION)
  {
    newPts->SetDataType(VTK_FLOAT);
  }
  else if (this->OutputPointsPrecision == vtkAlgorithm::DOUBLE_PRECISION)
  {
    newPts->SetDataType(VTK_DOUBLE);
  }

  newPts->SetNumberOfPoints(4 * numPts);
  newLines = vtkCellArray::New();
  newLines->AllocateEstimate(numPts * 3, 2);

  vtkNew<vtkUnsignedCharArray> cellColors;
  cellColors->SetNumberOfComponents(3);
  cellColors->SetName("Colors");
  cellColors->Allocate(numPts * 3);
  output->GetCellData()->SetScalars(cellColors);

  // Loop over all points, creating oriented line
  //
  for (ptId = 0; ptId < numPts; ptId++)
  {
    if (!(ptId % 10000)) // abort/progress
    {
      this->UpdateProgress(static_cast<double>(ptId) / numPts);
      if (this->GetAbortExecute())
      {
        break;
      }
    }

    input->GetPoint(ptId, x);
    if (this->VectorMode == VTK_USE_VECTOR)
    {
      inVectors->GetTuple(ptId, v);
    }
    else
    {
      inNormals->GetTuple(ptId, v);
    }
    for (i = 0; i < 3; i++)
    {
      newX[i] = x[i] + this->ScaleFactor * v[i];
    }

    double v_len = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    double dir[3] = {0.0, 0.0, 0.0};
    double ortho[3] = {0.0, 0.0, 0.0};
    if (v_len > 0) {
        dir[0] = v[0]/v_len;
        dir[1] = v[1]/v_len;
        dir[2] = v[2]/v_len;
        ortho[0] = -v[1]/v_len;
        ortho[1] = v[0]/v_len;
        ortho[2] = 0.0;
    }
    
    double arrow_len = this->ArrowheadSize;
    double arrow_wid = this->ArrowheadSize * 0.5;
    double p_a1[3], p_a2[3];
    for (i = 0; i < 3; i++) {
        p_a1[i] = newX[i] - arrow_len * dir[i] + arrow_wid * ortho[i];
        p_a2[i] = newX[i] - arrow_len * dir[i] - arrow_wid * ortho[i];
    }

    vtkIdType pts_main[2] = {ptId, ptId + numPts};
    vtkIdType pts_a1[2]   = {ptId + numPts, ptId + 2 * numPts};
    vtkIdType pts_a2[2]   = {ptId + numPts, ptId + 3 * numPts};

    newPts->SetPoint(pts_main[0], x);
    newPts->SetPoint(pts_main[1], newX);
    newPts->SetPoint(pts_a1[1], p_a1);
    newPts->SetPoint(pts_a2[1], p_a2);

    newLines->InsertNextCell(2, pts_main);
    cellColors->InsertNextTuple3(0, 0, 0); // Black main line

    newLines->InsertNextCell(2, pts_a1);
    cellColors->InsertNextTuple3(0, 0, 0); // Black arrowhead side 1

    newLines->InsertNextCell(2, pts_a2);
    cellColors->InsertNextTuple3(0, 0, 0); // Black arrowhead side 2

    outputPD->CopyData(pd, ptId, pts_main[0]);
    outputPD->CopyData(pd, ptId, pts_main[1]);
    outputPD->CopyData(pd, ptId, pts_a1[1]);
    outputPD->CopyData(pd, ptId, pts_a2[1]);
  }

  // Update ourselves and release memory
  //
  output->SetPoints(newPts);
  newPts->Delete();

  output->SetLines(newLines);
  newLines->Delete();

  return 1;
}

int vtkHedgeHogCustom::FillInputPortInformation(int, vtkInformation* info)
{
  info->Set(vtkAlgorithm::INPUT_REQUIRED_DATA_TYPE(), "vtkDataSet");
  return 1;
}

void vtkHedgeHogCustom::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  os << indent << "Scale Factor: " << this->ScaleFactor << "\n";
  os << indent << "Orient Mode: "
     << (this->VectorMode == VTK_USE_VECTOR ? "Orient by vector\n" : "Orient by normal\n");
  os << indent << "Output Points Precision: " << this->OutputPointsPrecision << "\n";
}
