param basename string
param location string

// for avoiding name collision
var uniqueSuffix = substring(uniqueString(resourceGroup().id), 0, 4)

module storage './storage.bicep' = {
  name: 'st${basename}${uniqueSuffix}deployment'
  params: {
    storageName: 'st${basename}${uniqueSuffix}'
    location: location
  }
}

module registry './registry.bicep' = {
  name: 'cr${basename}${uniqueSuffix}deployment'
  params: {
    registryName: 'cr${basename}${uniqueSuffix}'
    location: location
  }
}

module vault './keyvault.bicep' = {
  name: 'kv${basename}${uniqueSuffix}deployment'
  params: {
    vaultName: 'kv${basename}${uniqueSuffix}'
    location: location
  }
}

module insights './insights.bicep' = {
  name: 'ai${basename}${uniqueSuffix}deployment'
  params: {
    insightsName: 'ai${basename}${uniqueSuffix}'
    location: location
  }
}

resource aml 'Microsoft.MachineLearningServices/workspaces@2022-01-01-preview' = {
  name: 'aml${basename}${uniqueSuffix}'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: basename
    publicNetworkAccess: 'Enabled'
    storageAccount: storage.outputs.storageId
    containerRegistry: registry.outputs.registryId
    keyVault: vault.outputs.vaultId
    applicationInsights: insights.outputs.insightsId
  }
}

resource computeInstance 'Microsoft.MachineLearningServices/workspaces/computes@2022-01-01-preview' = {
  name: 'compute-${uniqueSuffix}'
  location: location
  parent: aml
  properties: {
    computeType: 'ComputeInstance'
    disableLocalAuth: true
    properties: {
      vmSize: 'Standard_NC6'
      applicationSharingPolicy: 'Personal'
      sshSettings: {
        sshPublicAccess:'Disabled'
      }
    }
  }
}

resource computeCluster 'Microsoft.MachineLearningServices/workspaces/computes@2022-01-01-preview' = {
  name: 'cluster-${uniqueSuffix}'
  location: location
  parent: aml
  properties: {
    computeType: 'AmlCompute'
    disableLocalAuth: true
    properties: {
      remoteLoginPortPublicAccess: 'NotSpecified'
      scaleSettings: {
        maxNodeCount: 2
        minNodeCount: 0
        nodeIdleTimeBeforeScaleDown: 'PT120S'
      }
      vmPriority: 'Dedicated'
      vmSize: 'Standard_NC24rs_v3'
    }
  }
}

output AML_NAME string = aml.name
output AML_ENDPOINT string = aml.properties.discoveryUrl
output AML_WORKSPACE string = aml.properties.workspaceId
output AML_STORAGE string = aml.properties.storageAccount
output AML_KEYVAULT string = aml.properties.keyVault
