#!/usr/bin/env ruby
# frozen_string_literal: true

ROOT = File.expand_path("..", __dir__)
WORKFLOW_PATH = File.join(ROOT, ".github/workflows/pages.yml")
VERIFY_PATH = File.join(ROOT, "scripts/verify_site.rb")

failures = []

unless File.file?(WORKFLOW_PATH)
  warn "Pages workflow check failed: .github/workflows/pages.yml is missing."
  exit 1
end

workflow = File.read(WORKFLOW_PATH)
verify_script = File.read(VERIFY_PATH)

def require_text(failures, source, text, message)
  failures << message unless source.include?(text)
end

require_text(failures, workflow, "push:\n    branches:\n      - master", "Pages workflow must run on pushes to master.")
require_text(failures, workflow, "pull_request:\n    branches:\n      - master", "Pages workflow must verify pull requests targeting master.")
require_text(failures, workflow, "run: npm run verify:site", "Pages workflow must run the complete site verification.")
require_text(failures, workflow, "JEKYLL_ENV: production", "Pages verification must use the production Jekyll environment.")
require_text(failures, workflow, "uses: actions/upload-pages-artifact@v4", "Pages workflow must upload the verified _site artifact.")
require_text(failures, workflow, "path: _site", "Pages artifact must be built from _site.")
require_text(failures, workflow, "uses: actions/deploy-pages@v5", "Pages workflow must deploy through the Pages deployment action.")
require_text(failures, workflow, "needs: build", "Deployment must depend on the verification/build job.")
require_text(failures, workflow, "pages: write", "Deployment needs Pages write permission.")
require_text(failures, workflow, "id-token: write", "Deployment needs an OIDC identity token.")
require_text(failures, workflow, "name: github-pages", "Deployment must use the protected github-pages environment.")

verify_position = workflow.index("run: npm run verify:site")
upload_position = workflow.index("uses: actions/upload-pages-artifact@v4")
deploy_position = workflow.index("uses: actions/deploy-pages@v5")

if verify_position && upload_position && verify_position > upload_position
  failures << "Complete verification must run before the Pages artifact is uploaded."
end

if upload_position && deploy_position && upload_position > deploy_position
  failures << "The verified Pages artifact must be uploaded before deployment."
end

required_static_checks = %w[
  scripts/check_reader_paths.rb
  scripts/check_bilingual_articles.rb
  scripts/check_wiki_index.rb
  scripts/check_manifesto_home.rb
]

required_static_checks.each do |check|
  failures << "verify:site must include #{check}." unless verify_script.include?(check)
end

if failures.any?
  warn "Pages workflow check failed:"
  failures.each { |failure| warn "- #{failure}" }
  exit 1
end

puts "Pages workflow check passed: verification gates artifact upload and deployment."
