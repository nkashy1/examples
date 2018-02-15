# Toxic comment classification with kubeflow

In this example, we will develop a neural model to estimate various kinds of
toxicity in Wikipedia comments. We will use this model to participate in
the [Toxic Comment Classification
Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
on Kaggle.

We will do everything - model design, training, evaluation, and generation of
submissions - on a [Google Container Engine](https://cloud.google.com/kubernetes-engine/)
(GKE) cluster running kubeflow components. For shared storage across pods, we
will use a [Google Cloud Storage](https://cloud.google.com/storage/) (GCS)
bucket.


## Prerequisites

This example assumes basic familiarity with
[kubeflow](https://github.com/kubeflow/kubeflow). If you are unfamiliar with it,
it is worth going through the [kubeflow user
guide](https://github.com/kubeflow/kubeflow/blob/master/user_guide.md). At the
very least, you should have
[`kubectl`](https://kubernetes.io/docs/tasks/tools/install-kubectl/) and
[`ksonnet`](https://ksonnet.io/) installed locally.

This example will also require you to interact with Google Cloud Platform. There
are generally two ways of doing this -- using the [Cloud
Console](https://console.cloud.google.com) or using the [Cloud
SDK](https://cloud.google.com/sdk/) which provides you with a suite of command
line tools (`gcloud` and `gsutil` being the relevant ones for this example). We
will show preference to the command line except in situations where something is
genuinely much easier to do through the Cloud Console UI.


## Spinning up a GKE cluster

We will now create a GKE cluster on which we will go through the workflow
presented in this example. To do so, we will require the following pieces of
information:

1. Name of cluster - we will use the name `toxicity` by default, but you can
   change this below.

1. Number of nodes in the cluster - By default, if this is not specified, GKE
   creates 3-node clusters. However, for our particular use case, we will only
   need 2 nodes. We specify this with the `--num-nodes 2` argument below.

1. Machine type used for the nodes in the cluster - A complete list of standard
   machine types is available as part of the [Google Compute Engine (GCE)
   documentation](https://cloud.google.com/compute/docs/machine-types). By
   default, GKE uses the `n1-standard-1` machine type to provision the nodes in
   a cluster. However, as we intend to perform some more processor-heavy tasks,
   let us use `--machine-type n1-standard-16`. You can vary this below depending
   on your requirements.

Putting it all together, we create our cluster as follows:

```bash
CLUSTER_NAME=toxicity
gcloud container clusters create $CLUSTER_NAME --num-nodes 2 --machine-type n1-standard-16
```

This can some time as GKE has to start up the underlying GCE instances and set
them up to expose the kubernetes API. When it is done, you should see something
like this as output:

```bash
NAME      LOCATION    MASTER_VERSION  MASTER_IP       MACHINE_TYPE    NODE_VERSION  NUM_NODES  STATUS
toxicity  us-west1-b  1.7.12-gke.1    35.185.219.215  n1-standard-16  1.7.12-gke.1  2          RUNNING
```

Once the cluster has been created, you can explicitly make it available to
your local `kubectl` by running

```bash
gcloud container clusters get-credentials $CLUSTER_NAME
```

Once you have run that command, the following command should display a nonempty
string:

```bash
kubectl config current-context | grep $CLUSTER_NAME
```

You should also see `--num-nodes` entries in the table that is displayed with

```bash
kubectl get node
```

Finally, you will have to designate yourself as a cluster administrator for the
GKE cluster. You can do this by running:

```bash
kubectl create clusterrolebinding default-admin --clusterrole=cluster-admin --user=$(gcloud config get-value account)
```

**Note:** Please do not forget to delete your cluster once you have run through
this example. Otherwise you *will* be charged for the idle time. You can do so
by running:

```bash
gcloud container clusters delete $CLUSTER_NAME
```


## Setting up a GCS bucket for shared storage across pods

First let us create the bucket using `gsutil`. By default, we will set the
bucket name to `toxicity-<PROJECT>` where `<PROJECT>` is the name of the GCP
project you are working under (note that this should be the same project in
which you created the GKE cluster above). You can change this as desired in the
following snippet:

```bash
BUCKET_NAME=gs://toxicity-$(gcloud config get-value project)
gsutil mb $BUCKET_NAME
```

Pods running on your GKE cluster will not immediately have access to this
bucket. We will provide them with access by injecting the appropriate
credentials into their environments.

First, we have to generate the credentials. We do this by creating a GCP service
account with the appropriate permissions and downloading credentials for that
service account.

### Creating a service account

To create a service account named `toxicity` (this can be changed by changing
the `SERVICE_ACCOUNT_NAME` variable below):

```bash
SERVICE_ACCOUNT_NAME=toxicity
gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME --display-name $SERVICE_ACCOUNT_NAME
```

A service account on GCP is identified by its e-mail address. You can assign
this to the `SERVICE_ACCOUNT_EMAIL` variable by running:

```bash
SERVICE_ACCOUNT_EMAIL=$(gcloud iam service-accounts list --format "table[no-heading](EMAIL)" --filter NAME=$SERVICE_ACCOUNT_NAME)
```

To verify that the variable was set with a nonempty string:

```bash
echo $SERVICE_ACCOUNT_EMAIL
```

### Granting permissions to the service account

The service account we created above still does not have the appropriate
permissions to read to and write from our GCS bucket. We will have to grant it
the appropriate IAM role to do so.

To view a list of all the available IAM roles, run

```bash
gcloud iam roles list
```

In this case, we want to grant our service account the
`roles/storage.objectAdmin` role as it should have the ability to read, write,
and delete objects from Google Cloud Storage. We grant this role by running:

```bash
gcloud projects add-iam-policy-binding $(gcloud config get-value project) --member serviceAccount:$SERVICE_ACCOUNT_EMAIL --role roles/storage.objectAdmin
```

### Generating service account credentials

Now that we have created the service account, we should download credentials for
that service account which we will register as a secret on our kubernets
cluster. We can download these credentials using:

```bash
gcloud iam service-accounts keys create sacred.json --iam-account $SERVICE_ACCOUNT_EMAIL
```

This will create a file called `sacred.json` in the location from which you ran
the command.

With the credentials generated, we must now deliver them to our kubernetes
cluster in a secure manner.

### Deploying service account credentials to kubernetes cluster

We will create a [kubernetes secret](https://kubernetes.io/docs/concepts/configuration/secret/)
to represent the service account credentials we just generated.

To do this, we use the `kubectl create secret` command. Assuming you are working
in the directory into which you stored the service account credentials (as
`sacred.json`) above, simply run:

```bash
kubectl create secret generic gcp-service-account --from-file=./sacred.json
```

With this, we can mount the `gcp-service-account` secret into our pods and have
them interact freely with GCS.


## Developing with kubeflow

Now we are ready to begin our model development.

```
ks init toxicity-kf
cd toxicity-kf
ks registry add kubeflow github.com/kubeflow/kubeflow/tree/master/kubeflow
ks pkg install kubeflow/core
ks pkg install kubeflow/tf-serving
ks pkg install kubeflow/tf-job
```

```
kubectl create namespace toxicity
ks generate core kubeflow-core --name kubeflow-core --namespace toxicity
```

```
ks env add toxicity-gke
ks param set kubeflow-core cloud gke --env toxicity-gke
```

```
ks show toxicity-gke -c kubeflow-core > kubeflow-core.yaml
```

Added the secret manually to `kubeflow-core.yaml`:

```
---
apiVersion: apps/v1beta1
kind: StatefulSet
metadata:
name: tf-hub
namespace: toxicity
spec:
  .
  .
  .
    spec:
      containers:
        - command:
          - jupyterhub
          - -f
          - /etc/config/jupyterhub_config.py
          image: gcr.io/kubeflow/jupyterhub-k8s:1.0.1
          name: tf-hub
          .
          .
          .
          env:
            - name: GOOGLE_APPLICATION_CREDENTIALS
              value: "etc/creds/sacred.json"
          serviceAccountName: jupyter-hub
          volumes:
            - configMap:
                name: jupyterhub-config
              name: config-volume
            - name: creds
              secret:
                secretName: gcp-service-account
  updateStrategy:
    .
    .
    .
```


