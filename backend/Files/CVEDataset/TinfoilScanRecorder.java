package com.tinfoilsecurity.plugins.tinfoilscan;

import java.io.IOException;

import javax.servlet.ServletException;

import org.apache.commons.lang.StringUtils;
import org.kohsuke.stapler.DataBoundConstructor;
import org.kohsuke.stapler.QueryParameter;
import org.kohsuke.stapler.StaplerRequest;

import com.tinfoilsecurity.api.Client;
import com.tinfoilsecurity.api.Client.APIException;

import hudson.EnvVars;
import hudson.Extension;
import hudson.Launcher;
import hudson.model.AbstractBuild;
import hudson.model.AbstractProject;
import hudson.model.BuildListener;
import hudson.model.Result;
import hudson.tasks.BuildStepDescriptor;
import hudson.tasks.BuildStepMonitor;
import hudson.tasks.Publisher;
import hudson.tasks.Recorder;
import hudson.util.FormValidation;
import net.sf.json.JSONException;
import net.sf.json.JSONObject;

public class TinfoilScanRecorder extends Recorder {

  private String  apiAccessKey;
  private String  apiSecretKey;
  private String  apiHost;
  private String  siteID;
  private String  proxyHost;
  private Integer proxyPort;

  // Fields in config.jelly must match the parameter names in the "DataBoundConstructor"
  @DataBoundConstructor
  public TinfoilScanRecorder(String accessKey, String secretKey, String apiHost, String siteID, String proxyHost,
      Integer proxyPort) {
    this.apiAccessKey = accessKey;
    this.apiSecretKey = secretKey;
    this.apiHost = apiHost;
    this.siteID = siteID;
    this.proxyHost = proxyHost;
    this.proxyPort = proxyPort;
  }

  public String getAPIAccessKey() {
    return apiAccessKey;
  }

  public String getAPISecretKey() {
    return apiSecretKey;
  }

  public String getAPIHost() {
    return apiHost;
  }

  public String getSiteID() {
    return siteID;
  }

  public String getProxyHost() {
    return proxyHost;
  }

  public Integer getProxyPort() {
    return proxyPort;
  }

  public BuildStepMonitor getRequiredMonitorService() {
    return BuildStepMonitor.STEP;
  }

  @Override
  public boolean perform(AbstractBuild<?, ?> build, Launcher launcher, BuildListener listener) {
    try {
      EnvVars environment = build.getEnvironment(listener);
      apiAccessKey = environment.expand(getAPIAccessKey());
      apiSecretKey = environment.expand(getAPISecretKey());

      Client tinfoilAPI = getDescriptor().buildClient(environment, apiAccessKey, apiSecretKey, getAPIHost(),
          getProxyHost(), getProxyPort());

      try {
        tinfoilAPI.startScan(siteID);

        String host = StringUtils.isNotBlank(getAPIHost()) ? getAPIHost() : getDescriptor().getAPIHost();

        listener.getLogger()
            .println("Tinfoil Security scan started! Log in to " + host + "/sites to view its progress.");
      }
      catch (APIException e) {
        listener.getLogger().println("Your Tinfoil Security scan could not be started. " + e.getMessage());
      }
      finally {
        tinfoilAPI.close();
      }

      build.setResult(Result.SUCCESS);
    }
    catch (InterruptedException e) {
      listener.getLogger().println("Your Tinfoil Security scan could not be started. " + e.getMessage());
    }
    catch (IOException e) {
      listener.getLogger().println("Your Tinfoil Security scan could not be started. " + e.getMessage());
    }
    return true;
  }

  // Overridden for better type safety.
  @Override
  public DescriptorImpl getDescriptor() {
    return DESCRIPTOR;
  }

  @Extension
  public static final DescriptorImpl DESCRIPTOR = new DescriptorImpl();

  public static class DescriptorImpl extends BuildStepDescriptor<Publisher> {

    private String  apiHost;
    private String  apiAccessKey;
    private String  apiSecretKey;
    private String  proxyHost;
    private Integer proxyPort;

    public DescriptorImpl() {
      load();
    }

    @Override
    public boolean isApplicable(Class<? extends AbstractProject> jobType) {
      // Applicable for all job types.
      return true;
    }

    @Override
    public String getDisplayName() {
      return "Tinfoil Security";
    }

    public String getDefaultAPIHost() {
      return Client.DEFAULT_API_HOST;
    }

    // This gets called when you save global settings. See global.jelly.
    @Override
    public boolean configure(StaplerRequest req, JSONObject json) throws FormException {
      apiHost = json.getString("apiHost");
      if (StringUtils.isBlank(apiHost)) {
        apiHost = getDefaultAPIHost();
      }
      apiAccessKey = json.getString("accessKey");
      apiSecretKey = json.getString("secretKey");
      proxyHost = json.getString("proxyHost");

      try {
        if (StringUtils.isBlank(proxyHost)) {
          proxyPort = null;
        }
        else {
          proxyPort = json.getInt("proxyPort");
        }
      }
      catch (JSONException e) {
        proxyPort = null;
      }
      save();

      return super.configure(req, json);
    }

    public String getAPIHost() {
      return apiHost;
    }

    public String getAPIAccessKey() {
      return apiAccessKey;
    }

    public String getAPISecretKey() {
      return apiSecretKey;
    }

    public String getProxyHost() {
      return proxyHost;
    }

    public Integer getProxyPort() {
      return proxyPort;
    }

    public FormValidation doCheckProxyPort(@QueryParameter String value, @QueryParameter String proxyHost)
        throws IOException, ServletException {
      if (StringUtils.isBlank(proxyHost)) {
        return FormValidation.ok();
      }

      if (StringUtils.isBlank(value)) {
        return FormValidation.error("Proxy Port is required when Proxy Host is specified");
      }

      try {
        Integer.parseInt(value);
        return FormValidation.ok();
      }
      catch (NumberFormatException e) {
        return FormValidation.error("Proxy Port must be a number");
      }
    }

    public Client buildClient(EnvVars environment, String apiAccessKey, String apiSecretKey, String apiHost,
        String proxyHost, Integer proxyPort) throws IOException, InterruptedException {

      if (StringUtils.isBlank(apiAccessKey)) {
        if (environment == null) {
          apiAccessKey = getAPIAccessKey();
        }
        else {
          environment.expand(getAPIAccessKey());
        }
      }
      if (StringUtils.isBlank(apiSecretKey)) {
        if (environment == null) {
          apiSecretKey = getAPISecretKey();
        }
        else {
          environment.expand(getAPISecretKey());
        }
      }

      Client client = new Client(apiAccessKey, apiSecretKey);

      if (StringUtils.isBlank(apiHost)) {
        apiHost = getAPIHost();
      }
      if (getDefaultAPIHost() != apiHost) {
        client.setAPIHost(apiHost);
      }

      if (!StringUtils.isBlank(proxyHost)) {
        client.setProxyConfig(proxyHost, proxyPort);
      }

      return client;
    }
  }
}
